# coding=utf-8
import json
import os
import glob

import datasets

from PIL import Image
import numpy as np

logger = datasets.logging.get_logger(__name__)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

class AihubConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(AihubConfig, self).__init__(**kwargs)

class AihubRawDataset(datasets.GeneratorBasedBuilder):
    """AIHUB Raw dataset."""

    BUILDER_CONFIGS = [
        AihubConfig(name="aihub", version=datasets.Version("1.0.0"), description="AIHUB dataset"),
    ]

    def __init__(self, *args, **kwargs):
        self.target_aihub_datasets = kwargs.pop('target_aihub_datasets', [])
        self.root_dir = kwargs.pop('root_dir', None)
        super().__init__(*args, **kwargs)
        
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC', '032_PUBLIC_ADMIN_DOCUMENT_OCR', '025_OCR_DATA_FINANCE_LOGISTICS']
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC']
        # self.target_aihub_datasets = ['032_PUBLIC_ADMIN_DOCUMENT_OCR']
        

    def _info(self):
        return datasets.DatasetInfo(
            description=None,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"root_dir": f"{self.root_dir}", "target_datasets": self.target_aihub_datasets, "type": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"root_dir": f"{self.root_dir}", "target_datasets": self.target_aihub_datasets, "type": "validation"}
            ),
        ]

    def get_file_paths_with_matching(self, root_dir, dataset_name, type="train"):
        ### 1. get images
        if dataset_name == '023_OCR_DATA_PUBLIC':
            if type == "train":
                img_glob_pattern = f'{root_dir}/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Training/01.원천데이터/**/*'
                ann_glob_pattern = f'{root_dir}/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Training/02.라벨링데이터/**/*'

            elif type == "validation":
                img_glob_pattern = f'{root_dir}/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Validation/01.원천데이터/**/*'
                ann_glob_pattern = f'{root_dir}/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*'

        elif dataset_name == '032_PUBLIC_ADMIN_DOCUMENT_OCR':
            if type == "train":
                img_glob_pattern = f'{root_dir}/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[[]원천[]]*/**/*'
                ann_glob_pattern = f'{root_dir}/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[[]라벨[]]*/**/*'

            elif type == "validation":
                img_glob_pattern = f'{root_dir}/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/02.Validation/[[]원천[]]*/**/*'
                ann_glob_pattern = f'{root_dir}/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/02.Validation/[[]라벨[]]*/**/*'
        
        elif dataset_name == '025_OCR_DATA_FINANCE_LOGISTICS':
            if type == "train":
                img_glob_pattern = f'{root_dir}/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Training/01.원천데이터/**/*'
                ann_glob_pattern = f'{root_dir}/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Training/02.라벨링데이터/**/*'
            
            elif type == "validation":
                img_glob_pattern = f'{root_dir}/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Validation/01.원천데이터/**/*'
                ann_glob_pattern = f'{root_dir}/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*'



        # img_dirs = sorted([d for d in os.listdir(img_root_dir) if os.path.isdir(os.path.join(img_root_dir, d))])
        # ann_dirs = sorted([d for d in os.listdir(ann_root_dir) if os.path.isdir(os.path.join(ann_root_dir, d))])
        
        ### filter ann, image files
        img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        ann_exts = ['.json', '.txt']

        img_files = []
        _img_files = glob.glob(img_glob_pattern, recursive=True)
        for _img_file in _img_files:
            if os.path.splitext(_img_file)[1].lower() in img_exts:
                img_files.append(_img_file)

        ann_files = []
        _ann_files = glob.glob(ann_glob_pattern, recursive=True)
        for _ann_file in _ann_files:
            if os.path.splitext(_ann_file)[1].lower() in ann_exts:
                ann_files.append(_ann_file)

        print('image and annotation file paths load completed!')

        ann_files_set = set(ann_files)

        found_img_files = []
        found_ann_files = []

        if dataset_name == '023_OCR_DATA_PUBLIC':
            ### convert list to set for matching data
            # /data/aihub/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Training/01.원천데이터/TS_OCR(public)_CST_2000_5280188_0002/CST_2000_5280188_0002_0599.jpg
            # /data/aihub/023_OCR_DATA_PUBLIC/01-1.정식개방데이터/Training/02.라벨링데이터/TL_OCR(public)_CST_2000_5280188_0002/CST_2000_5280188_0002_0599.json

            for img_file in img_files:
                img_file_split = img_file.split('/')
                img_file_split[-3] = '02.라벨링데이터'
                img_file_split[-2] = 'TL' + img_file_split[-2][2:] if img_file_split[-4] == 'Training' else 'VL' + img_file_split[-2][2:]
                img_file_split[-1] = os.path.splitext(img_file_split[-1])[0] + '.json'

                ann_file = '/'.join(img_file_split)
                
                if ann_file in ann_files_set:
                    found_img_files.append(img_file)
                    found_ann_files.append(ann_file)
                else:
                    print(f'{ann_file} 이 실제 ann 경로에 존재하지 않습니다.')

        elif dataset_name == '032_PUBLIC_ADMIN_DOCUMENT_OCR':
            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[원천]train_partly_labeling_16/2.원천데이터_부분라벨링/주민복지/5350030/1982/0001/5350030-1982-0001-0001.jpg
            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[라벨]train_partly_labling/주민복지/5350030/1982/0001/5350030-1982-0001-0001.json
            
            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[원천]train1/02.원천데이터(jpg)/인.허가/5350093/2001/5350093-2001-0001-0001.jpg
            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/01.Training/[라벨]train/01.라벨링데이터(Json)/인.허가/5350093/2001/5350093-2001-0001-0001.json

            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/02.Validation/[원천]validation/02.원천데이터(Jpg)/주민자치/5350047/1994/5350047-1994-0001-0707.jpg
            # /data/aihub/032_PUBLIC_ADMIN_DOCUMENT_OCR/01.데이터/02.Validation/[라벨]validation/01.라벨링데이터(Json)/주민자치/5350047/1994/5350047-1994-0001-0707.json

            for img_file in img_files:
                img_file_split = img_file.split('/')

                if 'train_partly_labeling' in img_file_split[-7]:
                    ann_file = '/'.join(img_file.split('/')[:-7]) + '/[라벨]train_partly_labling/' + '/'.join(img_file.split('/')[-5:])
                    ann_file = os.path.splitext(ann_file)[0] + '.json'

                elif 'train' in img_file_split[-6]:
                    ann_file = '/'.join(img_file.split('/')[:-6]) + '/[라벨]train/01.라벨링데이터(Json)/' + '/'.join(img_file.split('/')[-4:])
                    ann_file = os.path.splitext(ann_file)[0] + '.json'

                elif 'validation' in img_file_split[-6]:
                    ann_file = '/'.join(img_file.split('/')[:-6]) + '/[라벨]validation/01.라벨링데이터(Json)/' + '/'.join(img_file.split('/')[-4:])
                    ann_file = os.path.splitext(ann_file)[0] + '.json'

                else:
                    print(f'img file 경로 포맷이 맞지 않습니다. {img_file}')
                    continue

                if ann_file in ann_files_set:
                    found_img_files.append(img_file)
                    found_ann_files.append(ann_file)
                else:
                    print(f'{ann_file} 이 실제 ann 경로에 존재하지 않습니다.')

        elif dataset_name == '025_OCR_DATA_FINANCE_LOGISTICS':
            # /data/aihub/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_금융_1.은행_1-1.신고서/IMG_OCR_6_F_0031102.json
            # /data/aihub/025_OCR_DATA_FINANCE_LOGISTICS/01-1.정식개방데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서/IMG_OCR_6_F_0031102.png
            
            for img_file in img_files:
                img_file_split = img_file.split('/')
                img_file_split[-3] = '02.라벨링데이터'
                img_file_split[-2] = 'TL' + img_file_split[-2][2:] if img_file_split[-4] == 'Training' else 'VL' + img_file_split[-2][2:]
                img_file_split[-1] = os.path.splitext(img_file_split[-1])[0] + '.json'

                ann_file = '/'.join(img_file_split)
                
                if ann_file in ann_files_set:
                    found_img_files.append(img_file)
                    found_ann_files.append(ann_file)
                else:
                    print(f'{ann_file} 이 실제 ann 경로에 존재하지 않습니다.')
        print(f'{len(found_img_files)} files are found!')
        return found_img_files, found_ann_files
        
    
    def _generate_examples(self, root_dir, target_datasets, type):
        ### generate full file path with type and target_dataset
        ### get paths of img_files and ann_files
        print(target_datasets)
        for dataset_name in target_datasets:
            ### make ann_dir and img_dir for each datasets
            logger.info(f"⏳ Generating examples.. root_dir: {root_dir}, target_datasets: {target_datasets}")
            img_files, ann_files = self.get_file_paths_with_matching(root_dir, dataset_name, type)
            for guid, (img_file, ann_file) in enumerate(zip(img_files, ann_files)):
                words = []
                bboxes = []
                try:
                    with open(ann_file, "r", encoding="utf8") as f:
                        data = json.load(f)
                except:
                    print(f'{ann_file} can not be loaded!')
                    continue
                
                try:
                    image, size = load_image(img_file)
                except:
                    print(f'{img_file} can not be loaded!')
                    continue
                
                if dataset_name  == '023_OCR_DATA_PUBLIC':
                    json_bboxes = data['Bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)

                elif dataset_name == '032_PUBLIC_ADMIN_DOCUMENT_OCR':
                    json_bboxes = data['bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)

                elif dataset_name == '025_OCR_DATA_FINANCE_LOGISTICS':
                    json_bboxes = data['bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)
                    
                yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "image_path": img_file}


if __name__ == '__main__':
    aihub_raw_builder = AihubRawDataset()
    aihub_raw_builder.download_and_prepare()
    dataset = aihub_raw_builder.as_dataset(split='train')
    print(dataset)
    print(next(iter(dataset)))
    # first_batch = dataset['train'].take(1)

    # # 첫 번째 배치 출력
    # for sample in first_batch:
    #     print(sample)

    print('done!')