> 最近打算好好整理下之前保存过的关于Android的一些文章，网站，资源等，由于数量还是不少的，而且也会持续更新，所以会分成多篇文章。同时如果有好文章也希望能推荐给我，如果有链接失效的可以评论告诉我，谢谢。

---
##### 网络请求框架
1. [教你写Android网络框架之基本架构](http://www.devtf.cn/?p=662)
2. [Android 各大网络请求库的比较及实战](http://android.jobbole.com/81564/)
3. [Android网络请求心路历程](http://www.jianshu.com/p/3141d4e46240?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
4. [HTTP Status Code （http状态码）](http://www.cnblogs.com/yzenet/archive/2013/04/07/3003470.html)
5. [这是一个专用于解决Android中网络请求及图片加载的缓存处理框架](https://github.com/LittleFriendsGroup/KakaCache)
6. [Android推送技术研究](http://www.jianshu.com/p/584707554ed7#)
7. [Jsoup Java HTML Parser](https://jsoup.org/)(可以用来抓取网页数据并进行解析)
8. [打造属于自己的Android网络库](http://www.jianshu.com/p/16736df632a1)

---

###### [Volley](https://android.googlesource.com/platform/frameworks/volley)
> Volley的特点
> 
> - Volley的优势在于处理小文件的http请求；
> - 在Volley中也是可以使用Okhttp作为传输层
> - Volley在处理高分辨率的图像压缩上有很好的支持；
> - NetworkImageView在GC的使用模式上更加保守，在请求清理上也更加积极，networkimageview仅仅依赖于强大的内存引用，并当一个新请求是来自ImageView或ImageView离开屏幕时 会清理掉所有的请求数据。
> - 设计目标就是非常适合去进行数据量不大，但通信频繁的网络操作，而对于大数据量的网络操作，比如说下载文件等，Volley的表现就会非常糟糕


1. 【Android Volley完全解析系列--from 郭霖大神】
- [Android Volley完全解析(一)，初识Volley的基本用法](http://blog.csdn.net/guolin_blog/article/details/17482095)
- [Android Volley完全解析(二)，使用Volley加载网络图片](http://blog.csdn.net/guolin_blog/article/details/17482165)
- [Android Volley完全解析(三)，定制自己的Request](http://blog.csdn.net/guolin_blog/article/details/17612763)
- [Android Volley完全解析(四)，带你从源码的角度理解Volley](http://blog.csdn.net/guolin_blog/article/details/17656437)

2. [Android库Volley的使用介绍](http://bxbxbai.github.io/2014/09/14/android-working-with-volley/)
3. [使用OKHttp处理Volley的底层HTTP请求](http://willyan.me/2013/06/01/volley-okhttp/)

###### [android-async-http](https://github.com/loopj/android-async-http)

> 特点
> - 所以请求在子线程中完成，请求回调在调用该请求的线程中完成
> - 使用线程池
> - 使用RequestParams类封装请求参数
> - 支持文件上传
> - 持久化cookie到SharedPreferences，个人感觉这一点也是这个库的重要特点，可以很方便的完成一些模拟登录
> - 支持json
> - 支持HTTP Basic Auth

1. [快速Android开发系列网络篇之Android-Async-Http](http://www.cnblogs.com/angeldevil/p/3729808.html)
2. [Android网络请求库android-async-http使用](http://www.open-open.com/lib/view/open1392780943038.html)
3. [AsyncHttpClient 源码分析](http://www.cnblogs.com/xiaoweiz/p/3918042.html)
4. [android-async-http框架库源码走读](http://blog.csdn.net/yanbober/article/details/45307739)

###### [OkHttp](https://github.com/square/okhttp)
> 特点
> - OKHttp是Android版Http客户端。非常高效，支持SPDY、连接池、GZIP和 HTTP 缓存。
> - 默认情况下，OKHttp会自动处理常见的网络问题，像二次连接、SSL的握手问题。
> - 如果你的应用程序中集成了OKHttp，Retrofit默认会使用OKHttp处理其他网络层请求。
> - 从Android4.4开始HttpURLConnection的底层实现采用的是okHttp.

1. [OkHttp](http://square.github.io/okhttp/)
2. [OKHttp使用简介](http://blog.csdn.net/chenzujie/article/details/46994073)
3. [OkHttp使用教程](http://www.jcodecraeer.com/a/anzhuokaifa/androidkaifa/2015/0106/2275.html)
4. [高效地配置OkHttp](http://www.devtf.cn/?p=1264)
5. [Android OkHttp完全解析 是时候来了解OkHttp了](http://blog.csdn.net/lmj623565791/article/details/47911083)
6. [OkHttp源码解析](http://frodoking.github.io/2015/03/12/android-okhttp/)
7. 【OkHttp v2.4.0源码解析】系列
- [OKHttp源码解析（一）](http://blog.csdn.net/chenzujie/article/details/47061095)
- [OKHttp源码解析（二）](http://blog.csdn.net/chenzujie/article/details/47093723)
- [OKHttp源码解析（三）](http://blog.csdn.net/chenzujie/article/details/47158645)
8. 【OkHttp3源码分析】系列
- [OkHttp3源码分析[综述]](http://www.jianshu.com/p/aad5aacd79bf)
- [OkHttp3源码分析[复用连接池]](http://www.jianshu.com/p/92a61357164b)
- [OkHttp3源码分析[缓存策略]](http://www.jianshu.com/p/9cebbbd0eeab)
- [OkHttp3源码分析[DiskLruCache]](http://www.jianshu.com/p/23b8aa490a6b)
- [OkHttp3源码分析[任务队列])[http://www.jianshu.com/p/6637369d02e7)
9. [Android Https相关完全解析 当OkHttp遇到Https](http://blog.csdn.net/lmj623565791/article/details/48129405)


###### [Retrofit](https://github.com/square/retrofit)

> 特点
> - 性能最好，处理最快
> - 使用REST API时非常方便；
> - 传输层默认就使用OkHttp；
> - 支持NIO；
> - 拥有出色的API文档和社区支持
> - 速度上比volley更快；
> - 如果你的应用程序中集成了OKHttp，Retrofit默认会使用OKHttp处理其他网络层请求。
> - 默认使用Gson

1. [Retrofit](https://square.github.io/retrofit/)
2. [Retrofit2 完全解析 探索与okhttp之间的关系](http://blog.csdn.net/lmj623565791/article/details/51304204)
3. [你真的会用Retrofit2吗?Retrofit2完全教程](http://www.jianshu.com/p/308f3c54abdd?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
4. [Retrofit — Getting Started and Create an Android Client](https://futurestud.io/blog/retrofit-getting-started-and-android-client)
5. [好用的网络请求库Retrofit2（入门及讲解）](http://blog.csdn.net/biezhihua/article/details/49232289)
6. [Retrofit2与RxJava用法解析](http://www.cxbiao.com/2016/05/14/Retrofit2%E4%B8%8ERxJava%E7%94%A8%E6%B3%95%E8%A7%A3%E6%9E%90/)
7. [Retrofit分析-漂亮的解耦套路](http://www.jianshu.com/p/45cb536be2f4)
8. [RxJava 与 Retrofit 结合的最佳实践](http://gank.io/post/56e80c2c677659311bed9841)
9. [第六篇：网络请求篇（下）](http://www.jianshu.com/p/4c0b9793d0b7)
10. [Retrofit 2.0 + OkHttp 3.0 配置](https://drakeet.me/retrofit-2-0-okhttp-3-0-config)

---
###### JSON & Gson的使用
  在网络请求完毕后，一般请求得到的数据都是JSON格式，所以这里总结对JSON的解析方法。

1. [android 解析json数据格式](http://www.cnblogs.com/tt_mc/archive/2011/01/04/1925327.html)
2. [Android开发：JSON简介及最全面解析方法！](http://www.jianshu.com/p/b87fee2f7a23)
3. [Gson 2.4 使用指南系列]
- [你真的会用Gson吗?Gson使用指南（一）](http://www.jianshu.com/p/e740196225a4)
- [你真的会用Gson吗?Gson使用指南（二）](http://www.jianshu.com/p/c88260adaf5e)
- [你真的会用Gson吗?Gson使用指南（三）](http://www.jianshu.com/p/0e40a52c0063)
- [你真的会用Gson吗?Gson使用指南（四）](http://www.jianshu.com/p/3108f1e44155)
4. [Learning to Parse XML Data in Your Android App](https://www.sitepoint.com/learning-to-parse-xml-data-in-your-android-app/)


> 总结下：
> 
> 这4个网络请求框架中，有简单使用过`Volley`,`OkHttp`,`Retrofit`,而对`android-async-http`实际上是没有怎么了解过，主要也是因为现在非常流行使用的是`OkHttp+Retrofit`，然后`Volley`则是Google的新儿子，所以也是需要了解下。
> 
> 因为还是初学者，所以暂时给出的几个库的特点都是引用别人的分析，不过通过对这几个库的了解，还是可以看出`OkHttp+Retrofit`的强大，特别是结合现在一个非常热门的响应式编程`RxJava`的使用。



持续更新中...

