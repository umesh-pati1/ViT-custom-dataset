import datasets
from datasets.tasks import ImageClassification


_BASE_URL = "https://huggingface.co/datasets/umesh-pati1/chair/resolve/main/dataset.tar.gz"

_METADATA_URLS = {
    "test": "https://huggingface.co/datasets/umesh-pati1/chair/resolve/main/meta/test.txt",
    "train": "https://huggingface.co/datasets/umesh-pati1/chair/resolve/main/meta/train.txt",
}

_HOMEPAGE = "https://huggingface.co/datasets/umesh-pati1/chair"

_DESCRIPTION = (
    "This dataset consists of 1 chair category, with above 7000 images"
)

_CITATION = """\

"""

_LICENSE = """\
LICENSE AGREEMENT
=================
"""

_NAMES = [
    "Office Chairs",
]

_IMAGES_DIR = "dataset/images/"


class ExampleBuilder(datasets.GeneratorBasedBuilder):
    """Example Images dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            task_templates=[ImageClassification(
                image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(_BASE_URL)
        split_metadata_paths = dl_manager.download(_METADATA_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": dl_manager.iter_archive(archive_path),
                    "metadata_path": split_metadata_paths["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images": dl_manager.iter_archive(archive_path),
                    "metadata_path": split_metadata_paths["test"],
                },
            ),
        ]

    def _generate_examples(self, images, metadata_path):
        """Generate images and labels for splits."""
        with open(metadata_path, encoding="utf-8") as f:
            files_to_keep = set(f.read().split("\n"))
        for file_path, file_obj in images:
            if file_path.startswith(_IMAGES_DIR):
                if file_path[len(_IMAGES_DIR): -len(".jpg")] in files_to_keep:
                    label = file_path.split("/")[2]
                    yield file_path, {
                        "image": {"path": file_path, "bytes": file_obj.read()},
                        "label": label,
                    }
