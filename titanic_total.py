#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import wandb
import argparse
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

class TitanicDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.FloatTensor(X)
    self.y = torch.LongTensor(y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    feature = self.X[idx]
    target = self.y[idx]
    return {'input': feature, 'target': target}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return str


class TitanicTestDataset(Dataset):
  def __init__(self, X):
    self.X = torch.FloatTensor(X)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    feature = self.X[idx]
    return {'input': feature}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}".format(
      len(self.X), self.X.shape
    )
    return str


# __init__: 입력 데이터(특징 X, 타겟 y)를 파이토치가 사용하는 데이터 형식인 텐서(Tensor)로 변환합니다. 테스트 데이터셋은 y를 예측하는 것이 목표이므로 X만 가집니다.
# 
# __len__: 데이터셋에 있는 총 샘플 수를 반환합니다.
# 
# __getitem__: 인덱스(예: dataset[10])를 사용해 하나의 데이터 샘플(학습용은 특징과 타겟, 테스트용은 특징만)을 가져올 수 있게 합니다.
# 
# __str__: print() 함수로 출력될 때 데이터셋 크기와 형태에 대한 간단한 문자열 설명을 제공합니다.

# In[ ]:


def get_preprocessed_dataset():
    # 스크립트가 실행되는 현재 파일 경로를 기준으로 CSV 파일 경로 설정
    try:
        CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__이 정의되지 않은 환경(예: Jupyter)에서는 현재 작업 디렉터리를 사용
        CURRENT_FILE_PATH = os.getcwd()

    train_data_path = os.path.join(CURRENT_FILE_PATH, "train.csv")
    test_data_path = os.path.join(CURRENT_FILE_PATH, "test.csv")

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    all_df = pd.concat([train_df, test_df], sort=False)

    all_df = get_preprocessed_dataset_1(all_df)
    all_df = get_preprocessed_dataset_2(all_df)
    all_df = get_preprocessed_dataset_3(all_df)
    all_df = get_preprocessed_dataset_4(all_df)
    all_df = get_preprocessed_dataset_5(all_df)
    all_df = get_preprocessed_dataset_6(all_df)

    print("--- Preprocessed DataFrame Columns ---")
    print(all_df.columns)
    print("--- Preprocessed DataFrame Head ---")
    print(all_df.head(5))

    train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
    train_y = train_df["Survived"]

    test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

    print(f"\nInput Features ({len(train_X.columns)}): {train_X.columns.tolist()}")

    dataset = TitanicDataset(train_X.values, train_y.values)
    print("--- Full Train Dataset ---")
    print(dataset)

    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
    test_dataset = TitanicTestDataset(test_X.values)

    return train_dataset, validation_dataset, test_dataset


# CSV 찾기 & 로딩: 판다스를 이용해 train.csv와 test.csv 파일을 찾아 읽습니다.
# 
# 결합: 학습 데이터와 테스트 데이터를 하나로 합칩니다. 이렇게 하면 전처리 단계(결측치 채우기, 범주 인코딩 등)를 양쪽 데이터에 일관되게 적용할 수 있습니다.
# 
# 전처리: 보조 함수들(_1부터 _6까지)을 순서대로 호출하여 데이터를 정제하고 변환합니다.
# 
# 분리: 결합했던 데이터를 다시 학습용 특징(train_X), 학습용 레이블(train_y), 테스트용 특징(test_X)으로 나눕니다.
# 
# Dataset 생성: 사용자 정의 클래스(TitanicDataset, TitanicTestDataset)를 사용해 처리된 데이터를 감쌉니다.
# 
# 학습/검증 분할: 학습 데이터를 모델 훈련에 사용할 더 큰 세트와 훈련 중 성능 검증에 사용할 더 작은 세트로 나눕니다 (80/20 비율).
# 
# 반환: 파이토치의 DataLoader에서 바로 사용할 수 있는 최종 Dataset 객체들을 반환합니다.

# In[ ]:


def get_preprocessed_dataset_1(all_df):
    # Pclass별 Fare (요금) 평균값을 사용하여 Fare 결측치 메우기
    Fare_mean = all_df[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()
    Fare_mean.columns = ["Pclass", "Fare_mean"]
    all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
    all_df.loc[(all_df["Fare"].isnull()), "Fare"] = all_df["Fare_mean"]
    all_df = all_df.drop(columns=["Fare_mean"])
    return all_df


def get_preprocessed_dataset_2(all_df):
    # name을 세 개의 컬럼으로 분리하여 다시 all_df에 합침
    name_df = all_df["Name"].str.split("[,.]", n=2, expand=True)
    name_df.columns = ["family_name", "title", "name"]
    name_df["family_name"] = name_df["family_name"].str.strip()
    name_df["title"] = name_df["title"].str.strip()
    name_df["name"] = name_df["name"].str.strip()
    all_df = pd.concat([all_df, name_df], axis=1)
    return all_df


def get_preprocessed_dataset_3(all_df):
    # title별 Age 평균값을 사용하여 Age 결측치 메우기
    title_age_mean = all_df[["title", "Age"]].groupby("title").median().round().reset_index()
    title_age_mean.columns = ["title", "title_age_mean", ]
    all_df = pd.merge(all_df, title_age_mean, on="title", how="left")
    all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["title_age_mean"]
    all_df = all_df.drop(["title_age_mean"], axis=1)
    return all_df


def get_preprocessed_dataset_4(all_df):
    # 가족수(family_num) 컬럼 새롭게 추가
    all_df["family_num"] = all_df["Parch"] + all_df["SibSp"]
    # 혼자탑승(alone) 컬럼 새롭게 추가
    all_df.loc[all_df["family_num"] == 0, "alone"] = 1
    all_df["alone"].fillna(0, inplace=True)
    # 학습에 불필요한 컬럼 제거
    all_df = all_df.drop(["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"], axis=1)
    return all_df


def get_preprocessed_dataset_5(all_df):
    # title 값 개수 줄이기
    all_df.loc[
    ~(
            (all_df["title"] == "Mr") |
            (all_df["title"] == "Miss") |
            (all_df["title"] == "Mrs") |
            (all_df["title"] == "Master")
    ),
    "title"
    ] = "other"
    all_df["Embarked"].fillna("missing", inplace=True)
    return all_df


def get_preprocessed_dataset_6(all_df):
    # 카테고리 변수를 LabelEncoder를 사용하여 수치값으로 변경하기
    category_features = all_df.columns[all_df.dtypes == "object"]
    for category_feature in category_features:
        le = LabelEncoder()
        if all_df[category_feature].dtypes == "object":
          le = le.fit(all_df[category_feature])
          all_df[category_feature] = le.transform(all_df[category_feature])
    return all_df


# _1: 누락된 요금 값을 각 승객 등급의 평균 요금으로 채웁니다. 
# 
# _2: Name 열에서 호칭을 추출합니다. 
# 
# _3: 누락된 나이값을 추출된 title과 연관된 나이의 중앙값으로 채웁니다. 
# 
# _4: 새로운 특징인 family_num과 alone을 만듭니다. 모델링에 불필요하다고 판단되는 열을 제거합니다.
# 
# _5: 드문 호칭들을 'other'로 그룹화하여 title 열을 단순화합니다. 누락된 탑승 항구 값을 'missing'이라는 임시 값으로 채웁니다.
# 
# _6: 범주형 문자열 열을 Label Encoding을 사용해 숫자 표현으로 변환합니다. 

# In[ ]:


def get_data():
  # 1번 블록의 전처리 함수 호출 (이 함수는 다른 파일에 정의되어 있다고 가정)
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()

  print(f"\nTrain dataset size: {len(train_dataset)}")
  print(f"Validation dataset size: {len(validation_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")

  # wandb.config에서 배치 크기를 가져와 DataLoader 생성
  train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

  return train_data_loader, validation_data_loader, test_data_loader


# 데이터들을 DataLoader로 감싸서 모델 학습 시 데이터를 미니배치 단위로 효율적으로 공급할 수 있도록 합니다.

# In[ ]:


class MyModel(nn.Module):
  def __init__(self, n_input, n_output):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  # 입력 피처 10개 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, title, family_num, alone)
  # 출력 클래스 2개 (0: 사망, 1: 생존)
  my_model = MyModel(n_input=10, n_output=2)
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

  return my_model, optimizer


# __init__: 모델을 구성하는 층들을 정의합니다. 여기서는 입력층, 2개의 은닉층, 출력층으로 구성된 간단한 다층 퍼셉트론입니다. 각 층의 뉴런 수는 wandb.config에서 가져옵니다.
# 
# forward: 입력 데이터(x)가 모델의 층들을 어떤 순서로 통과하여 최종 출력을 만들어내는지 정의합니다

# In[ ]:


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = wandb.config.epochs
  loss_fn = nn.CrossEntropyLoss()  # 분류 문제이므로 CrossEntropyLoss 사용
  next_print_epoch = 100

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    correct_train = 0
    total_train = 0

    model.train() # 모델을 학습 모드로 설정
    for batch in train_data_loader:
      # Dataset이 딕셔너리 형태이므로 키로 접근
      input = batch['input']
      target = batch['target']

      output_train = model(input)
      loss = loss_fn(output_train, target)
      loss_train += loss.item()
      num_trains += 1

      # 정확도 계산
      _, predicted = torch.max(output_train.data, 1)
      total_train += target.size(0)
      correct_train += (predicted == target).sum().item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_validations = 0
    correct_validation = 0
    total_validation = 0

    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
      for batch in validation_data_loader:
        input = batch['input']
        target = batch['target']

        output_validation = model(input)
        loss = loss_fn(output_validation, target)
        loss_validation += loss.item()
        num_validations += 1

        # 정확도 계산
        _, predicted = torch.max(output_validation.data, 1)
        total_validation += target.size(0)
        correct_validation += (predicted == target).sum().item()

    train_accuracy = 100 * correct_train / total_train
    validation_accuracy = 100 * correct_validation / total_validation

    wandb.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations,
      "Training accuracy": train_accuracy,
      "Validation accuracy": validation_accuracy
    })

    if epoch % next_print_epoch == 0 or epoch == 1:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}, "
        f"Training Acc {train_accuracy:.2f}%, "
        f"Validation Acc {validation_accuracy:.2f}%"
      )
      if epoch >= next_print_epoch:
          next_print_epoch += 100


# 에포크 반복: 정해진 횟수만큼 전체 데이터셋 학습을 반복합니다.
# 
# 학습 모드: model.train()으로 모델을 학습 상태로 설정합니다.
# 
# 미니배치 학습: train_data_loader에서 데이터를 미니배치 단위로 가져와 다음을 수행합니다.
# 
# 모델 예측 (model(input))
# 
# 손실 계산 (loss_fn)
# 
# 역전파 (loss.backward())
# 
# 가중치 업데이트 (optimizer.step())
# 
# 학습 손실과 정확도 누적 계산
# 
# 평가 모드: model.eval()으로 모델을 평가 상태로 설정합니다 (드롭아웃 등 비활성화).
# 
# 검증: validation_data_loader에서 데이터를 가져와 모델 예측을 수행하고, 검증 손실과 정확도를 계산합니다 (가중치 업데이트는 안 함).
# 
# 로깅: 각 에포크의 학습/검증 손실과 정확도를 wandb에 기록합니다.
# 
# 출력: 주기적으로 학습 진행 상황(손실, 정확도)을 화면에 출력합니다.

# In[ ]:


def main(args):
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [20, 20], # 은닉층 설정은 그대로 사용
  }

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="titanic_survival_prediction", # wandb 프로젝트명 변경
    notes="Titanic survival prediction with MLP", # wandb 노트 변경
    tags=["mlp", "titanic"], # wandb 태그 변경
    name=current_time_str,
    config=config
  )
  print("--- wandb arguments ---")
  print(args)
  print("--- wandb config ---")
  print(wandb.config)

  # test_data_loader도 반환되지만, training_loop에서는 사용하지 않음
  train_data_loader, validation_data_loader, test_data_loader = get_data()

  linear_model, optimizer = get_model_and_optimizer()

  print("\n" + "#" * 50)
  print("Start Training...")
  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader
  )
  print("Training Finished.")
  print("#" * 50 + "\n")

  wandb.finish()


# 설정 로드: wandb 실험 설정(config)을 정의합니다 (에포크 수, 배치 크기 등).
# 
# wandb 초기화: 실험 추적을 위해 wandb를 설정하고 시작합니다. 프로젝트 이름, 노트, 태그 등을 지정합니다.
# 
# 데이터 로딩: get_data() 함수를 호출하여 학습/검증/테스트 데이터 로더를 가져옵니다.
# 
# 모델/옵티마이저 생성: get_model_and_optimizer() 함수를 호출하여 모델과 옵티마이저를 준비합니다.
# 
# 학습 시작: training_loop() 함수를 호출하여 모델 학습 및 검증을 시작합니다.
# 
# wandb 종료: 실험 기록을 마치고 wandb를 종료합니다.

# In[ ]:


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=16, help="Batch size (int, default: 16)" # 기본 배치 사이즈 16으로 변경
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=1_000, help="Number of training epochs (int, default:1_000)"
  )

  args = parser.parse_args()

  main(args)


# 터미널에서 실행할 때 --epochs, --batch_size 같은 인자를 받을 수 있게 설정합니다. 받은 인자를 main 함수에 전달하여 실행합니다.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




