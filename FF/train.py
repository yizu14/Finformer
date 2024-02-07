import os
import qlib
from qlib.workflow import R
from qlib.model.base import Model
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *
from metrics import *
from model import *
from loader import *
from audtorch.metrics.functional import concordance_cc

EXP_NAME = 'Finformer_exp'

'''Port Config'''
TOP_K = 30
N_DROP = 30


def metricsTester(ds):
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "EnhancedTopkDropoutStrategyBP",
            # Replace to the current path
            "module_path": "./metrics.py",
            "kwargs": {
                "signal": "<PRED>",
                "topk": TOP_K,
                "n_drop": N_DROP,
            },
        },
        "backtest": {
            "start_time": "2018-01-01",
            "end_time": "2022-12-31",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    with R.start(experiment_name="backtest_analysis"):
        global ba_rid
        global rid
        recorder = R.get_recorder(experiment_name=EXP_NAME)
        model = recorder.load_object("trained_model")

        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, ds, recorder)
        sar = SigAnaRecord(recorder)
        sr.generate()
        sar.generate()

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()


'''Finformer in Qlib Format'''


class Finformer_Model(Model):
    def __init__(self, d_feat, hidden_size, num_layers, temporal_dropout, snum_head, device):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_dropout = temporal_dropout
        self.device = device
        self.finformer = Finformer(d_feat=d_feat, hidden_size=hidden_size, temporal_dropout=temporal_dropout,
                                   snum_head=snum_head).to(device)

        for n in self.finformer.modules():
            if isinstance(n, nn.Linear):
                n.weight = nn.init.xavier_normal_(n.weight, gain=1.)

    def trainer(self, train_loader, optimizer):
        self.finformer.train()
        train_loader.train()

        loss_record = []
        batch_num = 0

        for data in train_loader:
            feature, label, index, daily_index = data['data'], data['label'], data['index'], data['daily_index']
            graph = graphReaderByDate(str(train_loader.restore_daily_index(daily_index).values).split('T')[0][2:]).to(
                self.device)
            batch_num += 1
            pred, mask = self.finformer(feature.float(), graph)
            optimizer.zero_grad()
            loss = -concordance_cc(pred, label)
            loss.backward()
            optimizer.step()
            loss_record.append(loss)

    def tester(self, epoch, test_loader, writer, prefix='Test'):
        self.finformer.eval()
        test_loader.eval()
        losses = []
        preds = []
        batch_num = 0
        itr = 0

        for data in test_loader:
            with torch.no_grad():
                feature, label, index, daily_index = data['data'], data['label'], data['index'], data['daily_index']
                graph = graphReaderByDate(
                    str(test_loader.restore_daily_index(daily_index).values).split('T')[0][2:]).to(self.device)
                batch_num += 1
                pred, mask = self.finformer(feature.float(), graph)
                loss = -concordance_cc(pred, label)
                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), },
                                          index=test_loader.restore_index(index)))
                itr += 1
                losses.append(loss.item())

        # evaluate
        preds = pd.concat(preds, axis=0)
        ic, rank_ic = metric_fn(preds)
        scores = ic

        writer.add_scalar(prefix + '/Loss', np.mean(losses), epoch)
        writer.add_scalar(prefix + '/std(Loss)', np.std(losses), epoch)
        writer.add_scalar(prefix + '/IC', np.mean(scores), epoch)
        writer.add_scalar(prefix + '/std(IC)', np.std(scores), epoch)

        return np.mean(losses), scores, ic, rank_ic

    def fit(self, dataset):

        train_loader, valid_loader, test_loader = create_mts_loader(dataset)

        save_path = "./Checkpoints"
        writer = SummaryWriter(log_dir=save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        optimizer = torch.optim.Adam(self.finformer.parameters(), lr=2e-4)
        stop_round = 0
        early_stop = 20
        n_epochs = 200
        best_score = -math.inf
        best_epoch = -1
        best_param = 0

        for epoch in tqdm(range(n_epochs)):
            pprint('Running Epoch:', epoch)
            pprint('training...')
            self.trainer(train_loader, optimizer)
            # train_loss, train_score, train_ic, train_rank_ic = self.tester(epoch, train_loader, criterion, writer, prefix='Train')
            valid_loss, valid_score, valid_ic, valid_rank_ic = self.tester(epoch, valid_loader, writer, prefix='Valid')
            test_loss, test_score, test_ic, test_rank_ic = self.tester(epoch, test_loader, writer, prefix='Test')

            pprint('valid_loss %.6f, test_loss %.6f' % (valid_loss, test_loss))
            pprint('valid_score %.6f, test_score %.6f' % (valid_score, test_score))
            pprint('valid_ic %.6f, test_ic %.6f' % (valid_ic, test_ic))
            pprint('valid_rank_ic %.6f, test_rank_ic %.6f' % (valid_rank_ic, test_rank_ic))

            if valid_score > best_score:
                best_score = valid_score
                best_param = copy.deepcopy(self.finformer.state_dict())
                best_epoch = epoch
                print('This is epoch {:.1f}, Saving FF with score {:.3f}...'.format(epoch, best_score))
                stop_round = 0

            else:
                stop_round += 1
                if stop_round >= early_stop:
                    pprint('early stop')
                    break
        pprint('best score:', best_score, '@', best_epoch)
        self.finformer.load_state_dict(best_param)
        torch.save(best_param, save_path + "/Stock_reg.ckpt")

    def predict(self, dataset):
        self.finformer.eval()
        batch_num = 0
        test_loader = dataset.prepare("test")
        preds = []
        pbar = tqdm(test_loader, position=0, leave=True)
        i = 0
        itr = 0

        for data in pbar:
            with torch.no_grad():
                feature, label, index, daily_index = data['data'], data['label'], data['index'], data['daily_index']
                graph = graphReaderByDate(
                    str(test_loader.restore_daily_index(daily_index).values).split('T')[0][2:]).to(self.device)
                batch_num += 1

                pred, mask = self.finformer(feature, graph)

                itr += 1
                X = np.c_[
                    pred.cpu().numpy(),
                ]
                pred = pd.DataFrame(X, index=index, columns=['score'])
                preds.append(pred)
                i += 1

        preds = pd.concat(preds, axis=0)
        preds.index = test_loader.restore_index(preds.index)
        preds.index = preds.index.swaplevel()
        preds.sort_index(inplace=True)

        return preds


'''Trainer'''


def trainer(ds, model_config):
    d_feat = model_config['d_feat']
    hidden_size = model_config['hidden_size']
    num_layers = model_config['num_layers']
    temporal_dropout = model_config['temporal_dropout']
    snum_head = model_config['snum_head']
    model = Finformer_Model(d_feat=d_feat,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            temporal_dropout=temporal_dropout,
                            snum_head=snum_head,
                            device="cuda:0")

    with R.start(experiment_name=EXP_NAME):
        model.fit(ds)
        R.save_objects(trained_model=model)
        global rid
        rec = R.get_recorder()
        rid = rec.id


if __name__ == '__main__':
    # Replace to the Qlib data path
    provider_uri = "/data/qlib_data/"
    qlib.init(provider_uri=provider_uri)
    date_config, model_config = yaml_reader("./config.yaml")
    device = 'cuda:0'
    dataset = getAlphaDataset(date_config)
    ds = getMTSDataset(dataset, date_config, device)
    trainer(ds, model_config)
    metricsTester(ds)
