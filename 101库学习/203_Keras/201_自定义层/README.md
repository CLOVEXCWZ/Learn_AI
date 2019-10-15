<font size=10>**自定义层**</font>



但是对于那些包含了可训练权重的自定义层，你应该自己实现这种层。在这种情况下，我们需要定义的是一个全新的、拥有可训练权重的层，这个时候我们就需要使用下面的方法。即通过编写自定义层，从Layer中继承。

这是一个 **Keras2.0** 中，Keras 层的骨架（如果你用的是旧的版本，请更新到新版）。你只需要实现三个方法即可:

要定制自己的层，**需要实现下面三个方法**

- build(input_shape)：这是定义权重的方法，可训练的权应该在这里被加入列表self.trainable_weights中。其他的属性还包括self.non_trainabe_weights（列表）和self.updates（需要更新的形如（tensor,new_tensor）的tuple的列表）。这个方法必须设置self.built = True，可通过调用super([layer],self).build()实现。
- call(x)：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量。
- compute_output_shape(input_shape)：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断。
   

