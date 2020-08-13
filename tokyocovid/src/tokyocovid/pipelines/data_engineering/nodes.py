# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd

# sample data: 1,130001,東京都,,2020-01-24,金,,湖北省武漢市,40代,男性,,,,,,1
def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
    data.rename(
        columns={
            '全国地方公共団体コード':'Code',
            '都道府県名':'PrefectureCt',
            '市区町村名':'PrefectureCt2',
            '公表_年月日':'Date',
            '曜日':'DoW',
            '発症_年月日':'OnsetDate',
            '患者_居住地':'Residence',
            '患者_年代':'AgeGroup',
            '患者_性別':'Gender',
            '患者_属性':'Attribute',
            '患者_状態':'Status',
            '患者_症状':'Symptom',
            '患者_渡航歴の有無フラグ':'TravelFlag',
            '備考':'Remarks',
            '退院済フラグ':'Discharged'
        }, inplace=True)

    # group by counts
    print(data.head())

    data = data[['Date','Gender','AgeGroup','Residence','PrefectureCt']]
    print(data.head())

    data = pd.get_dummies(data, columns=["Gender","AgeGroup","Residence"])
    data = data.groupby('Date').agg('count')

    print(data.tail())

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=False)   # why reset index ?

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * example_test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.drop(columns=['PrefectureCt'])
    train_data_y = training_data['PrefectureCt']
    test_data_x = test_data.drop(columns=['PrefectureCt'])
    test_data_y = test_data['PrefectureCt']

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )
