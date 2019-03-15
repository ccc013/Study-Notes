
第二篇文章：[设备的兼容性(Device Compatibility)](http://android.xsoftlab.net/guide/practices/compatibility.html)

### 设备的兼容性(Device Compatibility)
Android 现在可以运行在不同的设备上，不只是手机，还有平板电脑，电视。为了能让应用可以成功在这些设备上正常运行，应用应该允许一些特征功能的多样性并且可以提供一个灵活的用户界面来适应不同的屏幕配置。

为了达成这个目标，Android 提供一个可以让开发者使用一些静态文件(如对于不同屏幕尺寸的不同`XML`布局)来使用特定的配置的应用资源的动态应用框架。Android 可以根据当前设备的配置来加载合适的资源。所以对于开发者，在将你的程序打包成`apk`文件前，考虑好使用几套资源文件，比如使用几套不同大小的图片来适应不同的屏幕尺寸，对于手机和平板电脑要分别设置不同的布局文件等，这样当发布了你的应用后，就可以给使用不同设备的用户一个最好的用户体验。

最好的情况还是开发者可以指定其应用的功能要求，这样可以控制能够安装其应用的设备类型。更多有关让你的应用适应不同设备的内容，可以查看[Supporting Different Devices](http://android.xsoftlab.net/training/basics/supporting-devices/index.html)。

---
#### 1.兼容性意味着什么？
  当你阅读了越来越多有关 Android 开发的文章后，你可能在不同情况下会看到这个词语--`兼容性`。兼容性分为两种，一是设备的兼容性，二是应用的兼容性。
  
  因为 Android 是一个开源项目，所以任何硬件制造商都可以制造一个能运行 Android 系统的设备，但是一个设备被称为是"兼容 Android"只有在该设备上可以正常运行在 `Android 执行环境`下编写的应用的前提。而有关`Android 执行环境`的细节可以查看[ Android compatibility program](http://source.android.com/compatibility/overview.html),然后每个设备都必须通过兼容性测试套件(Compatibility Test Suite, CTS)才被认为是可兼容的。
  
  当然，作为一名应用开发者，是根本不需要考虑设备是否是兼容 Android的，因为可以包含谷歌应用商店(Google Play Store) 的设备都是可以兼容Android的，换句话说，能够安装你的应用的用户都是使用一台兼容Android的设备。
  
  那么，开发者需要考虑的就是你的应用的兼容性问题了。这也是因为 Android 可以运行在不同配置的设备上，有些功能不是所有的设备都具备的。比如一些设备是不具备有指南针传感器的，所以如果应该的核心功能需要有指南针传感器，那么就只有拥有这指南针传感器的设备可以使用你的应用了。

---
#### 2.控制应用对设备的可用性
  Android 支持很多特性功能，这些功能有些是需要硬件支持的，比如上述的指南针传感器，有些是基于软件的，比如应用程序部件，还有一些是依赖于平台的版本。不是每一种设备都支持所有的功能，所以开发者需要根据应用所要求的功能来控制应用对设备的可用性。
  
  为了让更多的用户可以使用你的应用，你应该在使用一个单独的`APK`的情况下支持尽可能多的设备配置。大多数情况下，你可以在运行的时候关闭可选的功能特性，然后根据不同配置使用不同资源文件(比如不同屏幕尺寸的布局文件，具体参考[提供应用资源](http://android.xsoftlab.net/guide/topics/resources/providing-resources.html))。如果可能，你是可以在谷歌应用商店中根据以下几种设备的特性来限制你的应用的可用性:
  
  * 设备功能(Device features)
  * 平台版本(Platform version)
  * 屏幕配置(Screen configuration)

---
##### (1) 设备功能(Device features)
  Android 给所有的硬件或软件功能都提供了功能`IDs`，比如对于指南针传感器的功能 ID 就是[`FEATURE_SENSOR_COMPASS`](http://android.xsoftlab.net/reference/android/content/pm/PackageManager.html#FEATURE_SENSOR_COMPASS),而应用程序组件的 ID 是[`FEATURE_APP_WIDGETS`](http://android.xsoftlab.net/reference/android/content/pm/PackageManager.html#FEATURE_APP_WIDGETS).
  
  在你的应用的[`manifest file`](http://android.xsoftlab.net/guide/topics/manifest/manifest-intro.html)可以通过使用[`<uses-feature>`](http://android.xsoftlab.net/guide/topics/manifest/uses-feature-element.html)来声明所需要的功能从而防止不具备应用所需要的功能的设备的用户安装你的应用。
  
  一个要求必须具备指南针传感器功能的应用可以如此写明该功能要求：
  
```
<manifest ... >
    <uses-feature android:name="android.hardware.sensor.compass"
                  android:required="true" />
    ...
</manifest>
```

  当然如果你的应用主要功能并不需要一个设备的功能，可以将上述代码中的`required`设置为`false`，然后在运行的时候检查设备的功能。如果应用的功能在当前设备上不可用的时候，那么可以关闭这项功能。一个检查当前设备是否具备某个功能的代码例子如下：
  
```
PackageManager pm = getPackageManager();
if (!pm.hasSystemFeature(PackageManager.FEATURE_SENSOR_COMPASS)) {
    // This device does not have a compass, turn off the compass feature
    disableCompassFeature();
}
```
这里使用到了[`hasSystemFeature()'](http://android.xsoftlab.net/reference/android/content/pm/PackageManager.html#hasSystemFeature(java.lang.String))这个函数，主要是检查设备是否支持某个功能，如果有返回`true`，否则返回`false`。

> 注意：有些[系统权限](http://android.xsoftlab.net/guide/topics/security/permissions.html)暗含着需要某些设备功能的支持。比如，如果你的应用需要蓝牙(`BLUETOOTH`)功能，它其实是需要有`FEATURE_BLUETOOTH`这个设备功能的。当然你也可以通过在`<uses-feature>`中对于的功能设置`required = false`来使得你的应用也可以运行在不具备蓝牙功能的设备上。更多的有关需要设备功能的权限的信息，可以查看[Permissions that Imply Feature Requirements](http://android.xsoftlab.net/guide/topics/manifest/uses-feature-element.html#permissions)。

---
##### (2) 平台版本(Platform version)
  不同的设备可能运行不同的系统版本，比如 Android 4.0 和 Android 4.4。而每个连续的系统版本都会增加一些前一个版本不可用的新的 APIs。每个系统版本都有指定一个 [`API level`](http://android.xsoftlab.net/guide/topics/manifest/uses-sdk-element.html#ApiLevels)。
  
  使用`API level`可以指定你的应用可以兼容的最低系统版本，需要使用`minSdkVersion`(在Android Studio中这个功能是放在`build.gradle`中了);同样也可以指定你的应用最适合使用的版本，使用`targetSdkVersion `这个属性。
  
  同样的可以在代码中动态地检查当前设备使用的系统版本是否能够支持应用的某些功能，代码例子如下
  
```
if (Build.VERSION.SDK_INT < Build.VERSION_CODES.HONEYCOMB) {
    // Running on something older than API level 11, so disable
    // the drag/drop features that use ClipboardManager APIs
    disableDragAndDrop();
}
```
上述例子是使用了[剪切板(ClipboardManager)](http://android.xsoftlab.net/reference/android/content/ClipboardManager.html)，这个API是在 API level 11中才增加的，所以低于这个系统版本的是无法实现这个功能的。

---
##### (3) 屏幕配置(Screen configuration)
  Android 可以运行在不同的屏幕尺寸上，包括手机，平板电脑以及电视。为了更好的根据屏幕类型来分类设备，Android 对每个设备都定义了两种特性: 屏幕尺寸(屏幕的物理尺寸）,以及屏幕密度(屏幕的物理密度，如`dpi`)。而为了简化这些不同的配置，Android 分别为这两种特性生成一些变量来方便使用：
  * 4种屏幕尺寸: `small`,`normal`,`large`,`xlarge`;
  * 几种密度: `mdpi(medium)`,`hdpi(hdpi)`,`xhdpi(extra high)`,`xxhdpi(extra-extra hdpi)`等。

  

---
#### 3. 总结
  这篇教程主要是说明 Android 的兼容性问题，主要是3个方面的兼容性问题，一是设备的功能，二是系统版本，三是屏幕配置。这些问题的产生原因当然是因为Android是一个开源的系统，所以很多手机制造商都可以运行 Android 的系统，但是这造成了有很多不同配置，不同屏幕大小的 Android 手机，所以这也是在开发过程必须考虑的一个问题，兼容性问题，当然个人感觉主要是最后一个问题会考虑得主要多点，就是屏幕配置问题，现在有很多尺寸的手机，不同分辨率的手机，在考虑界面的时候就需要多做几套图片，几个布局文件来适应尽可能多的不同屏幕配置的手机。
  
  关于屏幕适配的文章，这里推荐下最近看到的几篇文章，包括几位大神的文章：
  * 郭霖： [Android官方提供的支持不同屏幕大小的全部方法](http://blog.csdn.net/guolin_blog/article/details/8830286)
  * Stormzhang：[Android 屏幕适配](http://stormzhang.com/android/2014/05/16/android-screen-adaptation/)
  * 鸿洋：[Android 屏幕适配方案](http://blog.csdn.net/lmj623565791/article/details/45460089)
  * 凯子： [Android屏幕适配全攻略(最权威的官方适配指导)](http://blog.csdn.net/zhaokaiqiang1992/article/details/45419023)
  * Carson_Ho: [Android开发：最全面、最易懂的Android屏幕适配解决方案](http://www.jianshu.com/p/ec5a1a30694b)

  屏幕适配也是一个很重要的知识点，所以也是需要找时间好好看看这方面的内容。
  
  最后，如果有翻译不对，或者表达错误的地方，欢迎指正。