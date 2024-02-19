import torch
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as TF 
from sklearn.model_selection import train_test_split



def get_compose_list(img_size):
    '''
    get the compose function list based on img size
    '''
    assert len(img_size) == 3

    compose_list=[]

    if img_size[0] == 1:
        compose_list.append(TF.Grayscale(num_output_channels=img_size[0]))

    compose_list.extend(
        # every setting has the followings
        [TF.ToTensor(),
        TF.Resize((img_size[1],img_size[2]), 
                interpolation=TF.InterpolationMode.BICUBIC, 
                antialias=True),
        TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize to imagenet 
        ]
    )
    return compose_list


def load_one_image(image_path,transform = None):
    image = Image.open(image_path)
    image = image.convert("RGB")
    if transform is not None:
        image = transform(image)
    image = torch.unsqueeze(image,0)
    return image


def split(df,train_size=0.8,random_seed=42):
    '''
    Parameter:
    - df: dataFrame
    - train_size: proportion of train set
    - random_seed

    return:
    - df: df with one more column about split results
    '''
    df['malignant']=df['malignant'].astype(float)
    grouped = df.groupby(['skin_tone','malignant'])
    keys = grouped.groups.keys()
    
    train_index = []
    test_index = []

    for k in keys:
        subg = grouped.get_group(k)
        train, test = train_test_split(subg, random_state=random_seed,train_size=train_size)
        
        train_index.extend(train['DDI_ID'].to_list())
        test_index.extend(test['DDI_ID'].to_list())

    df['split'] = df.apply(lambda x: 'train' if x['DDI_ID'] in train_index else 'test', axis=1)
    return df


def loader(img_dir,
           default_split=False,
           random_seed = 42, 
           img_size = (3,224,224), # resize to
           device = None,
           ):
    '''
    loading (images,lab,sensitive_attributes) from the dir path img_dir

    parameter:
    - default_split: use the split from 'ddi_metadat_split.csv'
    - img_size: resize to

    return: (X_train, X_test, y_train, y_test, a_train, a_test)
    '''
    print('loading DDI dataset')

    if default_split:
        df_split = pd.read_csv('./datafiles/ddi_metadata_split.csv')
    else:
        csv_file_path = img_dir+'/ddi_metadata.csv'
        df = pd.read_csv(csv_file_path)
        df_split = split(df,random_seed=random_seed)


    transform = TF.Compose(get_compose_list(img_size))

    def get_data_from_subset(df_,):
        X, y, a = [],[],[]

        for index, row in df_.iterrows():
            image_filename = row['DDI_file'] 
            label = row['malignant'] 
            sensitive_attribute = row['skin_tone']

            image_path = os.path.join(img_dir, image_filename)
            X.append(load_one_image(image_path,transform))

            y.append(label)
            a.append(sensitive_attribute)
        
        X = torch.cat(X, dim=0).to(device)
        y = torch.tensor(y).to(device)
        a = torch.tensor(a).to(device)

        return X,y,a


    X_train, y_train, a_train = get_data_from_subset(df_split[df_split['split']=='train'])
    X_test, y_test, a_test = get_data_from_subset(df_split[df_split['split']=='test'])

    print(f'Finish loading.\n#train: {len(X_train)}\n#test: {len(X_test)}')

    return X_train, y_train, a_train, X_test, y_test, a_test