import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from typing import List


class OutliersData:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Outliers Class
        :param data: Initial Dataframe
        """
        self.data = data.copy()
        self.data = self.data[~self.data.isna()]

    def z_score_outlier(self) -> List:
        """
        The Z score method is a statistical technique used to determine how many standard deviations
        a data point is away from the mean. This method is useful in analyzing data and determining
        the probability of an event occurring. The Z score is calculated by subtracting the mean
        from the data point and then dividing the result by the standard deviation. The resulting
        number indicates the number of standard deviations the data point is away from the mean.
        :return: List of outliers
        """
        out = []
        if not any(self.data):
            return out
        m = np.mean(self.data)
        sd = np.std(self.data)
        for i in self.data:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        return out

    def zr_score_outlier(self) -> List:
        """
        The Robust Z-score method is an extension of the Z-score and MAD methods for outlier detection.
        It is also known as the "Median Absolute Deviation Z-score" method. Like the MAD method,
        it uses the median and absolute deviation from the median to determine how many median absolute
        deviations an observation is from the median. However, it also incorporates a correction
        factor based on the sample size and the kurtosis of the data, which improves its performance
        when the data is not normally distributed. This makes the Robust Z-score method a more robust
        technique for outlier detection than the standard Z-score method.
        :return: List of outliers
        """
        out = []
        if not any(self.data):
            return out
        med = np.median(self.data)
        ma = stats.median_abs_deviation(self.data)
        for i in self.data:
            z = (0.6745 * (i - med)) / (np.median(ma))
            if np.abs(z) > 3:
                out.append(i)
        return out

    def iqr_outliers(self) -> List:
        """
        The Interquartile Range (IQR) is a measure of the spread of a dataset that is less affected by
        outliers than the standard deviation. It is calculated as the difference between the third
        quartile and the first quartile. To detect outliers using the IQR method, any value that falls
        outside the range of -1.5 times IQR to 1.5 times IQR is considered an outlier. This range is
        also called the "inner fences" or "whiskers" of the box plot. The IQR method is a simple and
        robust way to identify outliers in a dataset.
        :return: List of outliers
        """
        out = []
        if not any(self.data):
            return out
        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower_tail = q1 - 1.5 * iqr
        upper_tail = q3 + 1.5 * iqr
        for i in self.data:
            if i > upper_tail or i < lower_tail:
                out.append(i)
        return out

    def winsorization_outliers(self) -> List:
        """
        The Winsorization method, also known as percentile capping, is a technique used to deal with
        outliers by limiting extreme values to a certain percentile. This method is similar to the
        IQR method, but instead of using the IQR to determine the range of acceptable values, it
        uses percentiles. In particular, any value that exceeds the value of the 99th percentile
        or falls below the 1st percentile of the data is replaced by the value at the 99th or 1st
        percentile, respectively. This effectively "caps" extreme values, reducing their impact
        on statistical analyses while preserving the overall distribution of the data.
        :return: List of outliers
        """
        out = []
        if not any(self.data):
            return out
        q1 = np.percentile(self.data, 1)
        q3 = np.percentile(self.data, 99)
        for i in self.data:
            if i > q3 or i < q1:
                out.append(i)
        return out

    def db_outliers(self) -> List:
        """
        DBSCAN is a density-based clustering algorithm used for identifying outliers in a dataset by
        dividing it into subgroups based on high-density regions. Outliers are identified as points
        that do not belong to any of the high-density regions or clusters. The algorithm assigns a
        cluster label to each data point, with cluster -1 indicating that the data point is an outlier.
        The DBSCAN approach is similar to the K-means clustering method, but it is more effective for
        multivariate outlier detection. The algorithm requires two parameters, namely, the minimum
        number of data points required to form a dense region and a distance metric that defines the
        neighborhood of a data point. Overall, DBSCAN is a robust technique for identifying outliers
        in large datasets with complex structures.
        :return: List of outliers
        """
        if not any(self.data):
            return []
        outlier_detection = DBSCAN(eps=2, metric="euclidean", min_samples=5)
        clusters = outlier_detection.fit_predict(self.data.values.reshape(-1, 1))

        data = pd.DataFrame()
        data["cluster"] = clusters
        outliers = data["cluster"].value_counts().sort_values(ascending=False)
        outliers = outliers.tolist()

        return list(set([self.data.iloc[x-1] for x in outliers]))

    def iso_outliers(self) -> List:
        """
        Isolation Forest is an outlier detection algorithm that belongs to the family of ensemble decision
        trees. It is similar to the Random Forest algorithm in principle. The algorithm works by randomly
        selecting a feature and a split value to create a decision tree. It then continues to create trees
        until it isolates an observation in a leaf node, i.e., the observation is the only one in the leaf
        node. The number of splits required to isolate the observation is an indicator of how different the
        observation is from the other observations in the dataset. The algorithm uses this information to
        identify outliers. Isolation Forest is a fast and efficient method for identifying outliers in
        high-dimensional datasets with complex structures.
        :return: List of outliers
        """
        if not any(self.data):
            return []
        iso = IsolationForest(random_state=1, contamination="auto")
        clusters = iso.fit_predict(self.data.values.reshape(-1, 1))

        data = pd.DataFrame()
        data["cluster"] = clusters
        outliers = data["cluster"].value_counts().sort_values(ascending=False)
        outliers = outliers.to_frame()["cluster"]

        return list(set([self.data.iloc[x] for x in outliers]))
