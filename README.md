## 1.1 Plants Data Set数据集理解

    Plants Data Set数据集包含了每一种植物(种类和科属)以及它们生长的地区。数据集中总共有69个地区，主要分布在美国和加拿大。一条数据(对应于文件中的一行)包含一种植物(或者某一科属)及其在上述69个地区中的分布情况。可以这样理解，该数据集中每一条数据包含两部分内容，如下图所示。

![image](https://upload-images.jianshu.io/upload_images/852606-e337185f21619d31.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

        例如一条数据:abronia fragrans,az,co,ks,mt,ne,nm,nd,ok,sd,tx,ut,wa,wy。其中abronia fragrans是植物名称(abronia是科属，fragrans是名称)，从az一直到wy是该植物的分布区域，采用缩写形式表示，如az代表的是美国Arizona州。植物名称和分布地区用逗号隔开，各地区之间也用逗号隔开。

## 1.2 数据预处理

        首先原数据集有30000多条数据集，如果直接对原数据集进行聚类操作，每聚类一次所消耗的时间比较长，本实验通过预处理部分，在对结果影响不大的情况下对数据进行了缩减。例如：

        1 abelmoschus,ct,dc,fl,hi,il,ky,la,md,mi,ms,nc,sc,va,pr,vi

        2 abelmoschus esculentus,ct,dc,fl,il,ky,la,md,mi,ms,nc,sc,va,pr,vi

        3 abelmoschus moschatus,hi,pr

        上述数据中第1行给出了所有属于abelmoschus这一科属的植物的分布地区，接下来的2、3两行分别列出了属于abelmoschus科属的两种具体植物及其分布地区。我们可以看出abelmoschus的分布情况已经包含了之后两种具体植物的分布情况，并且我们通过对数据二维的分布情况进行观察，可以用科属来代替具体植物来进行聚类分析，二维分布如下：

![image](https://upload-images.jianshu.io/upload_images/852606-a44db28b891e110d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

        同时该数据集由于地名是使用英文缩写来代替，这并不利于聚类，所以本实验将每条数据处理成以下数据形式。

0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   

        这是一个长度69的列表，每一位分别代表一个州，如果列表中的数字为1，则代表该植物在该州有分布，反之亦然。数据预处理之后的形式为3382*69大小的矩阵。

## 2.1 Sales_Transactions_Dataset_Weekly数据理解

        Sales_Transactions_Dataset_Weekly数据集是在52周之内，800多种商品每周的购买数量，同时本数据集也提供了标准化值。由于不同商品之间由于各自性质的不同，所以不能简单的通过件数来进行聚类，本实验我们采用了标准化值来进行聚类。通过聚类来找到商品之间是否存在内在的联系。首先本实验通过PCA降维将数据直观的显示在二维平面上，其分布情况如下图所示：

![image](https://upload-images.jianshu.io/upload_images/852606-3515b7b61f966ee6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 2.2 数据预处理

        本实验通过标准化值进行聚类，所以我们要对实验所给的CSV文件进行预处理，处理之后的数据存储在矩阵中，之后进行聚类操作。

## 3.1 Synthetic Control Chart Time Series数据理解

        本实验采用的实验数据集为Synthetic Control Chart Time Series，这个数据集经常被用于研究时间序列聚类算法中，数据集包含600个时间序列，每个时间序列由60个时间点构成,这组数据包含6类，每个类包含100个时间序列，这6类分别代表了一种时间序列变化趋势。

        1-100数据为标准时间序列

        101-200数据为周期型时间序列

        201-300数据为递增趋势的时间序列

        301-400数据为递减趋势的时间序列

        401-500数据为递增趋势时间序列，并且包含向上跳跃点

        501-600数据为递减趋势时间序列，并且包含向下跳跃点

![image](https://upload-images.jianshu.io/upload_images/852606-7f988e3b7a35146c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

        数据降维之后我们将直观的看到数据的分布情况，其中标准和周期、递增和递增向上、递减和递减向下三类之间分类比较明确，而类之间却非常接近，不易分别，以下实验就是探究聚类算法对于这几种数据的聚类效果。

![image](https://upload-images.jianshu.io/upload_images/852606-4edc69d120e22487.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 3.2  数据预处理

        由于本数据集为高维的时间序列，不能直接在二维坐标系上进行显示，也就无法直观的对数据的分布进行合理的观察，所以本实验将高维的时间序列进行了PCA降维预处理，使结果更加的直观。

## 4 实验部分

       本项目给出了K-means、DBscan、层次聚类在三个数据集上的代码，实验结果参考PDF文档部分。
