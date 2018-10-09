from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='SimHei'
import numpy as np


class Chimerge(object):

    def __init__(self,data):
        # 传入data，data为dateframe，格式为[feature_column,flag_column],flag取值0或1
        self.data=data
        self.feature_name,self.flag_name=data.columns
        value_counts = data[self.feature_name].value_counts()
        value_counts.sort_index(inplace=True)
        self.value_counts=value_counts
        self.value_sort = list(value_counts.index)

    def drop_min_value(self,min_rate=0.01):
        # 根据特征值大小排序，将特征值出现次数小于（总数据长度*min_rate）的特征值跟其前一个值或后一个值合并，默认min_rate等于0.01
        data_len=len(self.data)
        while min(self.value_counts) < data_len * min_rate:
            i = self.value_sort.index(self.value_counts.idxmin())
            if i == 0:
                j = i + 1
            elif i == len(self.value_counts) - 1:
                j = i - 1
            else:
                j = i - 1 if self.value_counts[self.value_sort[i - 1]] <= self.value_counts[self.value_sort[i + 1]] else i + 1

            if j < i:
                self.value_counts[self.value_sort[j]] += self.value_counts[self.value_sort[i]]
                self.value_counts.drop(self.value_sort[i], inplace=True)
                self.value_sort.pop(i)
            else:
                self.value_counts[self.value_sort[i]] += self.value_counts[self.value_sort[j]]
                self.value_counts.drop(self.value_sort[j], inplace=True)
                self.value_sort.pop(j)
        self.update_data()

    def update_data(self):
        # 根据新的value值，更新data
        data_copy = self.data.copy()
        for va in self.value_sort:
            bool_list = data_copy[self.feature_name] > va
            self.data[self.feature_name][bool_list] = va
            self.data[self.feature_name][data_copy[self.feature_name] < min(self.value_sort)] = min(self.value_sort)

    def ks_line(self):
        # 画ks曲线
        unstack_data = self.get_unstack_data()

        unstack_data.fillna(0, inplace=True)

        all_goo = unstack_data.loc[:, 0].sum()
        all_ba = unstack_data.loc[:, 1].sum()

        cum_unstack = unstack_data.cumsum()
        cum_unstack.columns = ['good', 'bad']
        plt.plot(cum_unstack.index, round(cum_unstack.good / all_goo, 3), color='green', label='累积0类')
        plt.plot(cum_unstack.index, round(cum_unstack.bad / all_ba, 3), color='red', label='累积1类')
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.legend()
        plt.show()
        plt.close()

    def get_unstack_data(self):
        group_data = self.data.groupby([self.feature_name, self.flag_name]).size()
        unstack_data = group_data.unstack()
        return unstack_data

    def chimerge(self,positive_weight=1.0,fillna_value=1,value_num=10):
        # chimerge分箱，这里是用0为positiv类，在样本不均衡情况下可以调节positive_rate,fillna_value用来填补空值，防止出现0值，value_num为目标分箱数
        good_rate = positive_weight
        unstack_data=self.get_unstack_data()
        unstack_data.iloc[:, 0] *= good_rate
        unstack_data.fillna(fillna_value, inplace=True)

        expect_num = value_num
        while len(unstack_data) > expect_num:
            kafang_lists = []
            for i in range(len(unstack_data) - 1):
                part_data = unstack_data.iloc[i:i + 2,:]
                total = part_data.iloc[:, 0].sum() + part_data.iloc[:, 1].sum()

                kafang = 0
                for c in range(2):
                    for r in range(2):
                        per_value = part_data.iloc[r, :].sum() * part_data.iloc[:, c].sum() / total
                        kafang += np.square(part_data.iloc[r, c] - per_value) / per_value

                kafang_lists.append(kafang)

            min_index = kafang_lists.index(min(kafang_lists))
            unstack_data.loc[self.value_sort[min_index], 0] += unstack_data.loc[self.value_sort[min_index + 1], 0]
            unstack_data.loc[self.value_sort[min_index], 1] += unstack_data.loc[self.value_sort[min_index + 1], 1]
            unstack_data.drop(self.value_sort[min_index + 1], inplace=True)
            self.value_counts.drop(self.value_sort[min_index+1],inplace=True)
            self.value_sort.pop(min_index + 1)
        self.update_data()



















