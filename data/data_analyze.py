import os

from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from data.registry import data_registry


class MyDataAnalyzer(DataAnalyzer):

    def __init__(
        self,
        dataset_class,
        device="cuda:7",
        worker=16,
        image_key="vol",
        label_key="lab",
    ):

        dataset_object = dataset_class(
            mode="train",
            fold_n=0,
            fold_size=5,
            rand_aug_type="none",  # can be anything
        )

        datapoints: list[dict] = dataset_object.parse()

        super().__init__(
            datalist={"training": datapoints},
            dataroot="",
            output_path=f"./data_stats/{dataset_class.__name__.lower()}.yaml",
            average=True,
            do_ccp=True,
            device=device,
            worker=worker,
            image_key=image_key,
            label_key=label_key,
        )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"

    dataset_names = ["Amos", "TotalSegmentatorOrgan", "Word"]

    for dataset_name in dataset_names:

        print(f"Analyzing {dataset_name}...")

        data_analyzer = MyDataAnalyzer(dataset_class=data_registry[dataset_name])
        data_analyzer.get_all_case_stats(key="training")
