
参考

- [How to release GPU memory after sess.close()? #19731](https://github.com/tensorflow/tensorflow/issues/19731)
- [how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory](https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory)
- [Tensorflow or cuda not giving back gpu memory after session closes #17048](https://github.com/tensorflow/tensorflow/issues/17048)



---
来自[How to release GPU memory after sess.close()? #19731](https://github.com/tensorflow/tensorflow/issues/19731)提及的方法：

1. 设置 tf.Session() 的配置参数
```
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
```


2. tf.Session. reset() 






