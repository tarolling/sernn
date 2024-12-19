import experiment_utils
from utils.jaxutils import key_iter
from utils.data import process_task


def main():
    cfg = experiment_utils.get_cfg("eeg")
    writer = experiment_utils.set_writer(cfg)

    key = key_iter(cfg["meta"]["seed"].get())

    task, (dataset, labels), (testset, testlabels), out_size = process_task(cfg)


if __name__ == "__main__":
    main()
