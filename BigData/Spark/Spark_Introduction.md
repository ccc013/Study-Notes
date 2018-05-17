# Spark 简介

------

### 1. 简介

​       Spark是加州大学伯克利分校 AMP 实验室（Algorithms, Machines, and People Lab）开
发**通用内存并行计算框架**。Spark 在 2013 年 6 月进入 Apache 成为孵化项目，8 个月后成为
Apache 顶级项目，速度之快足见过人之处，Spark 以其先进的设计理念，迅速成为社区的热门项目，围绕着 Spark 推出了 Spark SQL、 Spark Streaming、 MLLib 和 GraphX 等组件，也就是 BDAS（伯克利数据分析栈） ，这些组件逐渐形成**大数据处理一站式解决平台**。 

​      Spark 使用 **Scala** 语言进行实现，它是一种面向对象、函数式编程语言，能够像操作本地集合对象一样轻松地操作分布式数据集（Scala 提供一个称为 Actor 的并行模型，其中 Actor 通过它的收件箱来发送和接收非同步信息而不是共享数据，该方式被称为：**Shared Nothing** 模型） 。 在 Spark 官网上介绍，它具有下面几个特点：

* 运行速度快

​       **Spark 拥有 DAG（有向无环图） 执行引擎，支持在内存中对数据进行迭代计算。** 官方提供的数据表明，如果数据由磁盘读取，速度是 Hadoop MapReduce 的 10 倍以上，如果数据从内存中读取，速度可以高达 100 多倍 。

* 易用性好

​       Spark 不仅支持 Scala 语言，而且还支持 Java 和 Python 等语言编写应用程序，特别是   Scala 是一种高效、可拓展的语言，能够用简洁的代码处理较为复杂的处理工作。

* 通用性强

​    Spark 生态圈，也就是 BDAS (伯克利数据分析栈) 包含了 Spark Core、Spark SQL、Spark Streaming、MLLib 和 GraphX等组件，它们都是由 AMP 实验室提供，能够无缝的集成并提供一站式解决平台。

* 随处运行

​    Spark 具有很强的适应性，能够读取 HDFS、Cassandra、HBase、S3 和 Techyon 为持久层读写原生数据，能够以 Mesos、YARN 和自身携带的 Standalone 作为资源管理器调度 job，来完成 Spark 应用程序的计算。

### 2. Spark 和 Hadoop的差异

​    Spark 是在借鉴了 MapReduce 之上发展起来的，它继承了其分布式并行计算的优点并改进相应明显的缺点，具体如下：

（1）**Spark 把中间数据放到内存中，迭代运算效率高。**而 MapReduce 的计算结果需要保存到磁盘中，这极大降低了整体速度，而由于 **Spark 支持 DAG 图的分布式并行计算的编程框架，减少了迭代过程中数据的落地，提高了处理效率**。

（2）**Spark 容错性高。 Spark 引进了弹性分布式数据集 RDD （Resilient Distributed Dataset）的抽象，**它是分布在一组节点中的只读对象集合，这些集合是弹性的，如果数据集一部分丢失，则可以根据“血统”（即允许基于数据衍生过程）对它们进行重建。另外在 RDD 计算时可以通过 CheckPoint 来实现容错，而 CheckPoint 有两种方式：CheckPoint Data 和 Logging the Updates，用户可以控制采用哪种方式来实现容错。

（3）**Spark 更加通用**。不像 Hadoop 只提供了 Map 和 Reduce 两种操作，Spark 提供的不像 Hadoop 只提供了 Map 和 Reduce 两种操作，**Spark 提供的数据集操作类型有很多种，大致分为：Transformations 和 Actions 两大类。** Transformations 包括 Map、 Filter、 FlatMap、 Sample、 GroupByKey、 ReduceByKey、 Union、 Join、 Cogroup、MapValues、 Sort 和 PartionBy 等多种操作类型，同时还提供 Count, Actions 包括 Collect、Reduce、 Lookup 和 Save 等操作。 **另外各个处理节点之间的通信模型不再像 Hadoop 只有Shuffle 一种模式，用户可以命名、 物化，控制中间结果的存储、分区等。** 

### 3. 适用场景

​	目前大数据处理场景有以下几个类型： 

1. 复杂的批量处理（Batch Data Processing），偏重点在于处理海量数据的能力，至于处理
   速度可忍受，通常的时间可能是在数十分钟到数小时；
2. 基于历史数据的交互式查询（Interactive Query），通常的时间在数十秒到数十分钟之间
3. 基于实时数据流的数据处理（Streaming Data Processing），通常在数百毫秒到数秒之间 


​       目前对以上三种场景需求都有比较成熟的处理框架，第一种情况可以用 Hadoop 的MapReduce 来进行批量海量数据处理，第二种情况可以 Impala 进行交互式查询，对于第三种情况可以用 Storm 分布式处理框架处理实时流式数据。 以上三者都是比较独立，各自一套维护成本比较高，而 Spark 的出现能够一站式平台满意以上需求。 

​	通过以上分析，总结 Spark 场景有以下几个：

* Spark 是基于内存的迭代计算框架，**适用于需要多次操作特定数据集的应用场合**。需要反复操作的次数越多，所需读取的数据量越大，受益越大，数据量小但是计算密集度较大的场合，受益就相对较小 。

* 由于 RDD 的特性，Spark 不适用那种**异步细粒度更新状态的应用**，例如 web 服务的存储或者是增量的 web 爬虫和索引，就是对于那种增量修改的应用模型不适合 。

* 数据量不是特别大，但要求**实时统计分析需求**。


### 4. 


































