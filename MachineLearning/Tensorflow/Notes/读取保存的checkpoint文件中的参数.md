
参考：

- [How do I find the variable names and values that are saved in a checkpoint?](http://stackoverflow.com/questions/38218174/how-can-find-the-variable-names-that-saved-in-tensorflow-checkpoint/38226516#38226516)

---

如何从保存好的 model 文件中读取权重的名字和其数值，代码如下：

```
from tensorflow.python import pywrap_tensorflow
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names
```

