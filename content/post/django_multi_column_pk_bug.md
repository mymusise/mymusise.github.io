---
title: "Django使用联合主键的BUG"
date: 2017-12-16T23:19:14+08:00
draft: true
draft: false
tags: ["Django", "pk", "BUG"]
categories: ["系统"]

comment: false
toc: false
# reward: false
mathjax: false
---


# 0x01

记得在几个月前，项目中发现在使用Django ORM时候，调用`object.delete()`时候，把相关的其他几条也删除掉，DEBUG一段时间后还没找到问题，后来绕过这个坑用`Model.objects.filter().delete()`. 直到最近项目运行了一段时间，又发现了同样的问题 = =, 查了半天发现是Django现在不支持联合主键。具体情况如下：


# 0x02

有一个Model，用来保存系统用户和学生的关系，代码如下：
```python
class LeaderStudent(models.Model):
    user = models.OneToOneField(User, db_column="user_id", primary_key=True)
    student_info = models.OneToOneField(
        'StudentInfo', db_column="student_id", to_field='student_id', primary_key=True, )
    role = models.CharField(max_length=16, default='')

    class Meta:
        unique_together = (('user', 'student_info'),)
```
这个Model看起来像是使用了一个联合主键(`user`和`student_info`)，实际上在调用`object.pk`的时候，只会选取其中一个，我看开个django的shell可以看下：
```
In [1]: from apps.student.models import LeaderStudent
   ...: 
   ...: relation = LeaderStudent.objects.all().last()
   ...: relation.pk
   ...: 
Out[1]: 313

In [2]: print(relation._meta.pk)
student.LeaderStudent.user
```
所以实际上我们在调用`save()` `delete()`这些方法的时候，转成sql的时候用的`pk`只有一个。比如上面如果调用`relation.save()`， 转成sql就会是 `update table set ... where user_id = 313`。如果我们这个表里`user_id=313`有多列，那么这些都会被更新掉，就出现“莫名其妙”删掉了很多或者更新了好多列数据。

实际上我们只需要更新其中一条，先说一下解决方案： 
我们可以“曲线救国”： `LeaderStudent.objects.filter(user_id=313, student_info_id=student_id).update(...)`


# 0x03

从Django源码(1.11.x)上看了下，在 [django/db/models/options.py](https://github.com/django/django/blob/stable/1.11.x/django/db/models/options.py#L278)
```python
@python_2_unicode_compatible
class Options(object):
    def add_field(self, field, private=False, virtual=NOT_PROVIDED):
        ...
        if private:
            self.private_fields.append(field)
        elif field.is_relation and field.many_to_many:
            self.local_many_to_many.insert(bisect(self.local_many_to_many, field), field)
        else:
            self.local_fields.insert(bisect(self.local_fields, field), field)
            self.setup_pk(field)
        ...

    def setup_pk(self, field):
        if not self.pk and field.primary_key:
            self.pk = field
            field.serialize = False
```
熟悉Django ORM的话应该知道每个Model都有一个对应的`_meta`属性，每个`object.pk`调用的都是`object._meta.pk`对应的`field`的值（具体可以看下`Model._get_pk_val`这个方法）。在上面的代码里面可以看到，每次 `add_field`时候除了`private`和`many_to_many`，其他field都要`self.setup_pk(field)`. 从`setup_pk`函数可以看出，即使我们定义了两个`PK`，最后我们`PK`只会是第一个定义了`primary_key`的`field`。所以，像我们一开始的那个Model的这种写法，`unique_together`只能当唯一所以使用，`PK`还是只有一个的。


# 0x04

上面提到一种解决方案，直接绕开这个坑，filter出来然后用`QuerySet`去做`update`和`delete`来代替用object来操作。

另外，在Django's bug tracker里面也挂着这个问题 [https://code.djangoproject.com/wiki/MultipleColumnPrimaryKeys](https://code.djangoproject.com/wiki/MultipleColumnPrimaryKeys)。Django目前来说是不支持多个field来做`PK`的，不过里面提到有一个第三方的库`django-compositepks`可以实现多个field来组合成`PK`，但是这个库已经有3年多没有维护，支持好像也只到1.6版本，所以不是很推荐。

另外一种解决方案就是：添加另外一个自然序列的PK，原来的联合主键就当是唯一索引来使用了，像一开始那个Model，改下下面这样：
```python
class LeaderStudent(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.OneToOneField(User, db_column="user_id")
    student_info = models.OneToOneField(
        'StudentInfo', db_column="student_id", to_field='student_id')
    role = models.CharField(max_length=16, default='')

    class Meta:
        unique_together = (('user', 'student_info'),)
```
