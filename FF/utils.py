import datetime
import yaml
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm
from torch.utils.data import Sampler
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from loader import *


def yaml_reader(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data['date_config'], data['FF']


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


def getAlphaDataset(date_config):
    dh = Alpha360(instruments='csi300jq', start_time=date_config["START_TIME"], end_time=date_config["END_TIME"],
                  fit_start_time=date_config["FIT_START"],
                  fit_end_time=date_config["FIT_END"],
                  infer_processors=[
                      RobustZScoreNorm(fit_start_time=date_config["FIT_START"], fit_end_time=date_config["FIT_END"],
                                       fields_group="feature",
                                       clip_outlier="true"),
                      Fillna(fields_group="feature"),
                  ],
                  learn_processors=[
                      DropnaLabel(),
                      CSRankNorm(fields_group="label")
                  ],
                  label=(["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL"]))

    return dh


def edgeIndexTransform(adj):
    tmp_coo = sp.coo_matrix(adj)
    indices = np.vstack((tmp_coo.row, tmp_coo.col))
    edge_index = torch.LongTensor(indices)
    edge_index = to_undirected(edge_index)
    return edge_index


def graphReaderByDate(date):
    # Replace to the static relationship path
    csv = pd.read_csv("/data/industry_data/csi300/" + date + ".csv", index_col=['Unnamed: 0'])
    graph = edgeIndexTransform(csv)
    return graph


def pprint(*args):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)
