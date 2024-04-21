import os, sys
import glob

def get_file_paths(root_dir, dataset_name, type="train", validate=False):
    ### 1. get images
    ### files: {root_dir}/{dataset_dir}/img_train.txt, img_validation.txt, ann_train.txt, ann_validation.txt
    dataset_dir = f'{root_dir}/{dataset_name}'
    img_list_path = f'{dataset_dir}/img_{type}.txt'
    ann_list_path = f'{dataset_dir}/ann_{type}.txt'

    if dataset_name == '023_OCR_DATA_PUBLIC':
        if type == "train":
            img_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Training/01.원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Training/02.라벨링데이터/**/*'

        elif type == "validation":
            img_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Validation/01.원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*'

    elif dataset_name == '032_PUBLIC_ADMIN_DOCUMENT_OCR':
        if type == "train":
            img_glob_pattern = f'{dataset_dir}/01.데이터/01.Training/[[]원천[]]*/**/*'
            ann_glob_pattern = f'{dataset_dir}/01.데이터/01.Training/[[]라벨[]]*/**/*'

        elif type == "validation":
            img_glob_pattern = f'{dataset_dir}/01.데이터/02.Validation/[[]원천[]]*/**/*'
            ann_glob_pattern = f'{dataset_dir}/01.데이터/02.Validation/[[]라벨[]]*/**/*'
    
    elif dataset_name == '025_OCR_DATA_FINANCE_LOGISTICS':
        if type == "train":
            img_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Training/01.원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Training/02.라벨링데이터/**/*'
        
        elif type == "validation":
            img_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Validation/01.원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*'

    elif dataset_name == '055_FINANCE_OCR_DATA':
        if type == "train":
            img_glob_pattern = f'{dataset_dir}/055.금융업 특화 문서 OCR 데이터/01.데이터/1. Training/원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/055.금융업 특화 문서 OCR 데이터/01.데이터/1. Training/라벨링데이터/**/*'
        elif type == "validation":
            img_glob_pattern = f'{dataset_dir}/055.금융업 특화 문서 OCR 데이터/01.데이터/2. Validation/원천데이터/**/*'
            ann_glob_pattern = f'{dataset_dir}/055.금융업 특화 문서 OCR 데이터/01.데이터/2. Validation/라벨링데이터/**/*'

    # img_dirs = sorted([d for d in os.listdir(img_root_dir) if os.path.isdir(os.path.join(img_root_dir, d))])
    # ann_dirs = sorted([d for d in os.listdir(ann_root_dir) if os.path.isdir(os.path.join(ann_root_dir, d))])
    
    ### filter ann, image files
    img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
    ann_exts = ['.json', '.txt']


    if not os.path.exists(img_list_path):
        img_files = []
        _img_files = glob.glob(img_glob_pattern, recursive=True)
        for _img_file in _img_files:
            if os.path.splitext(_img_file)[1].lower() in img_exts:
                img_files.append(_img_file)

        # with open(img_list_path, 'w') as f:
        #     f.writelines(img_files)

    else:
        with open(img_list_path, 'r') as f:
            img_files = f.readlines()
            img_files = [x.strip() for x in img_files]

    if not os.path.exists(ann_list_path):
        ann_files = []
        _ann_files = glob.glob(ann_glob_pattern, recursive=True)
        for _ann_file in _ann_files:
            if os.path.splitext(_ann_file)[1].lower() in ann_exts:
                ann_files.append(_ann_file)

        # with open(ann_list_path, 'w') as f:
        #     f.writelines(ann_files)

    else:
        with open(ann_list_path, 'r') as f:
            ann_files = f.readlines()
            ann_files = [x.strip() for x in ann_files]
    
    print('image and annotation file paths load completed!')
    
    ### validate == True 이거나, list_path 파일이 없을 경우 validate 수행
    if validate or not os.path.exists(ann_list_path) or not os.path.exists(img_list_path):
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
            # /mnt/nas-drive-workspace/Datasets/aihub/025.OCR_데이터(금융_및_물류)/01-1.정식개방데이터/Training/01.원천데이터/TS_물류_5.기타_ET03/IMG_OCR_6_T_ET_016807.png
            
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
        
        elif dataset_name == '055_FINANCE_OCR_DATA':
            # /mnt/nas-drive-workspace/Datasets/aihub/055_FINANCE_OCR_DATA/055.금융업 특화 문서 OCR 데이터/01.데이터/1. Training/라벨링데이터/TL1/result/bank/annotations
            # /mnt/nas-drive-workspace/Datasets/aihub/055_FINANCE_OCR_DATA/055.금융업 특화 문서 OCR 데이터/01.데이터/1. Training/원천데이터/TS1/result/bank/images
            # /mnt/nas-drive-workspace/Datasets/aihub/055_FINANCE_OCR_DATA/055.금융업 특화 문서 OCR 데이터/01.데이터/2. Validation/라벨링데이터/VL1/result/bank/annotations
            for img_file in img_files:
                img_file_split = img_file.split('/')
                img_file_split[-6] = '라벨링데이터'
                img_file_split[-5] = 'TL' + img_file_split[-5][2:] if img_file_split[-7] == '1. Training' else 'VL' + img_file_split[-5][2:]
                img_file_split[-2] = 'annotations'
                img_file_split[-1] = os.path.splitext(img_file_split[-1])[0] + '.json'

                ann_file = '/'.join(img_file_split)
                
                if ann_file in ann_files_set:
                    found_img_files.append(img_file)
                    found_ann_files.append(ann_file)
                else:
                    print(f'{ann_file} 이 실제 ann 경로에 존재하지 않습니다.')

        if not os.path.exists(img_list_path):
            with open(img_list_path, 'w') as f:
                f.writelines([x + '\n' for x in found_img_files])
        
        if not os.path.exists(ann_list_path):
            with open(ann_list_path, 'w') as f:
                f.writelines([x + '\n' for x in found_ann_files])

        return found_img_files, found_ann_files
    
    else:
        return img_files, ann_files
    
if __name__ == '__main__':
    # dataset_name_list = ['023_OCR_DATA_PUBLIC', '032_PUBLIC_ADMIN_DOCUMENT_OCR', '025_OCR_DATA_FINANCE_LOGISTICS']
    dataset_name_list = ['023_OCR_DATA_PUBLIC']
    root_dir = '/mnt/nas-drive-workspace/Datasets/aihub/'
    dataset_name = dataset_name_list[0]
    get_file_paths(root_dir, dataset_name, type="train", validate=False)