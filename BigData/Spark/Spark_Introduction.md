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


### 4. Spark 术语

##### 运行模式

* **Local --本地模式**：常用于本地开发测试，本地还分为 local 单线程和 local-cluster 多线程；
* **Standalone -- 集群模式**：典型的 Master/slave 模式，不过也能看出 Master 是有单点故障的；Spark 支持 ZooKeeper 来实现 HA；
* **On yarn -- 集群模式**：运行在 yarn 资源管理器框架之上，由 yarn 负责资源管理，Spark 负责任务调度和计算；
* **On mesos -- 集群模式**：运行在 mesos 资源管理器框架之上，由 mesos 负责资源管理，Spark 负责任务调度和计算；
* **On cloud -- 集群模式**：比如 AWS（亚马逊）的 EC2，使用这个模式能很方便的访问 Amazon 的 S3；Spark 支持多种分布式存储系统：HDFS 和 S3。

##### 常用术语

![](http://7xrluf.com1.z0.glb.clouddn.com/spark%E6%9C%AF%E8%AF%AD.png)



### 5. 生态系统

​    Spark 生态圈是以 **Spark Core 为核心**，从 HDFS、Amazon S3 和 HBase 等持久层读取数据，以 MESS、YARN 和自身携带的 Standalone 为资源管理器调度 Job 完成 Spark 应用程序的计算，这些应用程序可以来自于不同的组件，如 Spark Shell / Spark Submit 的批处理、Spark Streaming 的实时处理应用、Spark SQL 的即席查询、BlinkDB 的权衡查询、MLlib / MLbase 的机器学习、GraphX 的图处理和 SparkR 的数学计算等。其生态系统如下图所示：

![](http://7xrluf.com1.z0.glb.clouddn.com/spark%E7%94%9F%E6%80%81%E7%B3%BB%E7%BB%9F.png)

接下来简单介绍各个组件。

##### Spark Core

总结下 Spark 内核架构：

* 提供了**有向无环图（DAG）**的分布式并行计算框架，并提供 **Cache 机制**来支持多次迭
  代计算或者数据共享，**大大减少迭代计算之间读取数据局的开销**，这对于需要进行多次
  迭代的数据挖掘和分析性能有很大提升；

* 在 Spark 中引入了 **RDD (Resilient Distributed Dataset)** 的抽象，它是分布在一组节
  点中的只读对象集合，这些集合是弹性的，如果数据集一部分丢失，则可以根据“ 血统”
  对它们进行重建，**保证了数据的高容错性**； 

* **移动计算而非移动数据**，RDD Partition 可以**就近读取分布式文件系统中的数据块**到各
  个节点内存中进行计算 ；

* 使用多线程池模型来减少 task 启动开销；

* 采用容错的、高可伸缩性的 akka 作为通讯框架。


##### SparkStreaming

​       SparkStreaming 是有个对实时数据流进行高通量、容错处理的流式处理系统，可以对多种数据源（如 Kdfka、Flume、Twitter、Zero 和 TCP 套接字）进行类似 Map、Reduce 和 Join 等复杂操作，并将结果保存到外部文件系统、数据库或应用到实时仪表盘。

​       SparkStreaming的构架如下：

* 计算流程：**Spark Streaming 是将流式计算分解成一系列短小的批处理作业。**这里的批处理引擎是 Spark Core，也就是把 Spark Streaming 的输入数据按照 batch size（如 1 秒）分成一段一段的数据（Discretized Stream），**每一段数据都转换成 Spark 中的 RDD（Resilient Distributed Dataset），**然后将 Spark Streaming 中对 DStream 的 Transformation 操作变为针对 Spark 中对 RDD 的 Transformation 操作，将 RDD 经过操作变成中间结果保存在内存中。整个流式计算根据业务的需求可以对中间的结果进行叠加或者存储到外部设备。 下图显示了 Spark Streaming 的整个流程。 

![](http://7xrluf.com1.z0.glb.clouddn.com/SparkStreaming%E8%AE%A1%E7%AE%97%E6%B5%81%E7%A8%8B.png)

* **容错性：**对于流式计算来说，容错性至关重要。首先我们要明确一下 Spark 中 RDD 的容错机制。**每一个 RDD 都是一个不可变的分布式可重算的数据集，其记录着确定性的操作继承关系（lineage）**，所以只要输入数据是可容错的，那么任意一个 RDD 的分区（Partition）出错或不可用，都是可以利用原始输入数据通过转换操作而重新算出的。 
* **实时性：**对于实时性的讨论，会牵涉到流式处理框架的应用场景。 Spark Streaming 将流式计算分解成多个 Spark Job，对于每一段数据的处理都会经过 Spark DAG 图分解以及 Spark的任务集的调度过程。**对于目前版本的 Spark Streaming 而言，其最小的 Batch Size 的选取在 0.5~2 秒钟之间（Storm 目前最小的延迟是 100ms 左右）**，所以 Spark Streaming 能够满足除对实时性要求非常高（如高频实时交易）之外的所有流式准实时计算场景。 
* **扩展性与吞吐量：**Spark 目前在 EC2 上已能够线性扩展到 100 个节点（每个节点 4Core），可以以数秒的延迟处理 6GB/s 的数据量（60M records/s），其吞吐量也比流行的 Storm 高2～5 倍，下图是 Berkeley 利用 WordCount 和 Grep 两个用例所做的测试，在 Grep 这个测试中，Spark Streaming 中的每个节点的吞吐量是 670k records/s，而 Storm 是 115krecords/s。 

![](http://7xrluf.com1.z0.glb.clouddn.com/Spark%20Streaming%20%E4%B8%8E%20Storm%20%E5%90%9E%E5%90%90%E9%87%8F%E6%AF%94%E8%BE%83%E5%9B%BE.png)

##### Spark SQL

Spark SQL 的特点:

* 引 入 了 新 的 RDD 类 型 SchemaRDD ， 可 以 象 传 统 数 据 库 定 义 表 一 样 来 定 义SchemaRDD，SchemaRDD 由定义了列数据类型的行对象构成。 SchemaRDD 可以从RDD 转换过来，也可以从 Parquet 文件读入，也可以使用 HiveQL 从 Hive 中获取。
* 内嵌了 Catalyst 查询优化框架，在把 SQL 解析成逻辑执行计划之后，利用 Catalyst 包里的一些类和接口，执行了一些简单的执行计划优化，最后变成 RDD 的计算
* 在应用程序中可以混合使用不同来源的数据，如可以将来自 HiveQL 的数据和来自 SQL的数据进行 Join 操作 

sparkSQL 在下面几点做了优化：

1. **内存列存储（In-Memory Columnar Storage）** sparkSQL 的表数据在内存中存储不是采用原生态的 JVM 对象存储方式，而是采用内存列存储；
2. **字节码生成技术（Bytecode Generation）** Spark1.1.0 在 Catalyst 模块的 expressions增加了 codegen 模块，使用动态字节码生成技术，对匹配的表达式采用特定的代码动态编译。 另外对 SQL 表达式都作了 CG 优化， CG 优化的实现主要还是依靠 Scala2.10 的运行时放射机制（runtime reflection） ；
3. **Scala 代码优化** SparkSQL 在使用 Scala 编写代码的时候，尽量避免低效的、容易 GC 的代码；尽管增加了编写代码的难度，但对于用户来说接口统一 


##### BlinkDB

​        **BlinkDB 是一个用于在海量数据上运行交互式 SQL 查询的大规模并行查询引擎**， 它允许用户通过权衡数据精度来提升查询响应时间，其数据的精度被控制在允许的误差范围内。为了达到这个目标，BlinkDB 使用两个核心思想:

* 一个**自适应优化框架**，从原始数据随着时间的推移建立并维护一组多维样本；
* 一个**动态样本选择策略**，选择一个适当大小的示例基于查询的准确性和（或）响应时间需求。 

和传统关系型数据库不同，BlinkDB 是一个很有意思的交互式查询系统，就像一个跷跷板，**用户需要在查询精度和查询时间上做一权衡**；如果用户想更快地获取查询结果，那么将牺牲查询结果的精度；同样的，用户如果想获取更高精度的查询结果，就需要牺牲查询响应时间。用户可以在查询的时候定义一个失误边界。 

##### ![](http://7xrluf.com1.z0.glb.clouddn.com/BlinkDB.png)

##### MLBase/MLlib

​        MLBase 是 Spark 生态圈的一部分专注于机器学习，让机器学习的门槛更低，让一些可能并不了解机器学习的用户也能方便地使用 MLbase。 MLBase 分为四部分：**MLlib、 MLI、 MLOptimizer 和 MLRuntime**。 

* ML Optimizer 会选择它认为最适合的已经在内部实现好了的机器学习算法和相关参数，来处理用户输入的数据，并返回模型或别的帮助分析的结果； 

* MLI 是一个进行特征抽取和高级 ML 编程抽象的算法实现的 API 或平台； 

* MLlib 是 Spark 实现一些常见的机器学习算法和实用程序，包括分类、 回归、 聚类、 协同过滤、 降维以及底层优化，该算法可以进行可扩充；

* MLRuntime 基于 Spark 计算框架，将 Spark 的分布式计算应用到机器学习领域。 


总的来说，**MLBase 的核心是他的优化器，把声明式的 Task 转化成复杂的学习计划，产出最优的模型和计算结果。**与其他机器学习 Weka 和 Mahout 不同的是：

* MLBase 是分布式的，Weka 是一个单机的系统；
* MLBase 是自动化的，Weka 和 Mahout 都需要使用者具备机器学习技能，来选择自己想要的算法和参数来做处理；
* MLBase 提供了不同抽象程度的接口，让算法可以扩充
* MLBase 基于 Spark 这个平台 



##### GraphX

​        GraphX 是 Spark 中用于**图**(e.g., Web-Graphs and Social Networks)和**图并行计算**(e.g.,PageRank and Collaborative Filtering)的 API,可以认为是 GraphLab(C++)和 Pregel(C++)在 Spark(Scala)上的重写及优化，跟其他分布式图计算框架相比，**GraphX 最大的贡献是，在Spark 之上提供一栈式数据解决方案，可以方便且高效地完成图计算的一整套流水作业。** 

​        **GraphX 的核心抽象是 Resilient Distributed Property Graph，一种点和边都带属性的有向多重图**。它扩展了 Spark RDD 的抽象，有 Table 和 Graph 两种视图，而只需要一份物理存储。两种视图都有自己独有的操作符，从而获得了灵活操作和执行效率。如同 Spark，GraphX的代码非常简洁。 GraphX 的核心代码只有 3 千多行，而在此之上实现的 Pregel 模型，只要短短的 20 多行。 GraphX 的代码结构整体下图所示，**其中大部分的实现，都是围绕 Partition 的优化进行的**。这在某种程度上说明了点分割的存储和相应的计算优化的确是图计算框架的重点和难点。 

![](http://7xrluf.com1.z0.glb.clouddn.com/GraphX%E7%BB%93%E6%9E%84%E5%9B%BE.png)

GraphX 的底层设计有以下几个关键点。

1. **对 Graph 视图的所有操作，最终都会转换成其关联的 Table 视图的 RDD 操作来完成**。这样对一个图的计算，最终在逻辑上，等价于一系列 RDD 的转换过程。因此，**Graph 最终具备了 RDD 的 3 个关键特性：Immutable、 Distributed 和 Fault-Tolerant。**其中最关键的是Immutable（不变性）。逻辑上，所有图的转换和操作都产生了一个新图；物理上，GraphX 会有一定程度的不变顶点和边的复用优化，对用户透明。
2. **两种视图底层共用的物理数据，由 RDD[Vertex-Partition]和 RDD[EdgePartition]这两个RDD 组 成** 。 点 和 边 实 际 都 不 是 以 表 Collection[tuple] 的 形 式 存 储 的 ， 而 是 由VertexPartition/EdgePartition 在内部存储一个带索引结构的分片数据块，以加速不同视图下的遍历速度。不变的索引结构在 RDD 转换过程中是共用的，降低了计算和存储开销。
3. **图的分布式存储采用点分割模式，而且使用 partitionBy 方法，由用户指定不同的划分策略（PartitionStrategy）**。划分策略会将边分配到各个 EdgePartition，顶点 Master 分配到各个 VertexPartition，EdgePartition 也会缓存本地边关联点的 Ghost 副本。划分策略的不同会影响到所需要缓存的 Ghost 副本数量，以及每个 EdgePartition 分配的边的均衡程度，需要根据图的结构特征选取最佳策略。目前有 EdgePartition2d、 EdgePartition1d、RandomVertexCut 和 CanonicalRandomVertexCut 这四种策略。在淘宝大部分场景下，EdgePartition2d 效果最好。 


##### SparkR

​       SparkR 是 AMPLab 发布的一个 R 开发包，使得 R 摆脱单机运行的命运，可以作为 Spark 的 job 运行在集群上，极大得扩展了 R 的数据处理能力。 

​      SparkR 的几个特性：

* 提供了 Spark 中弹性分布式数据集（RDD）的 API，用户可以在集群上通过 R shell 交互性的运行 Spark job。
* 支持序化闭包功能，可以将用户定义函数中所引用到的变量自动序化发送到集群中其他的机器上。
* SparkR 还可以很容易地调用 R 开发包，只需要在集群上执行操作前用 includePackage读取 R 开发包就可以了，当然集群上要安装 R 开发包 

##### Tachyon

​       Tachyon 是一个高容错的分布式文件系统，允许文件以内存的速度在集群框架中进行可靠的共享，就像 Spark 和 MapReduce 那样。通过利用信息继承，内存侵入，Tachyon 获得了高性能。 Tachyon 工作集文件缓存在内存中，并且让不同的 Jobs/Queries 以及框架都能内存的速度来访问缓存文件”。因此，Tachyon 可以减少那些需要经常使用的数据集通过访问磁盘来获得的次数。 Tachyon 兼容 Hadoop，现有的 Spark 和 MR 程序不需要任何修改而运行。
​	在 2013 年 4 月，AMPLab 共享了其 Tachyon 0.2.0 Alpha 版本的 Tachyon，其宣称性能为 HDFS 的 300 倍，继而受到了极大的关注。 Tachyon 的几个特性如下： 

* **JAVA-Like File API** Tachyon 提供类似 JAVA File 类的 API。

* **兼容性** Tachyon 实现了 HDFS 接口，所以 Spark 和 MR 程序不需要任何修改即可运行。

* **可插拔的底层文件系统** Tachyon 是一个可插拔的底层文件系统，提供容错功能。 tachyon 将内存数据记录在底层文件系统。它有一个通用的接口，使得可以很容易的插入到不同的底层文件系统。 目前支持HDFS，S3，GlusterFS 和单节点的本地文件系统，以后将支持更多的文件系统。 



