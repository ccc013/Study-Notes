package com.example.cai.gankcamp.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.widget.ImageView;

import com.bumptech.glide.Glide;
import com.example.cai.gankcamp.R;

import java.io.File;

/**
 * Created by： cai
 * Date：2016/5/1.
 */
public class GlideUtil {

    public static final String ANDROID_RESOURCE = "android.resource://";
    public static final String FOREWARD_SLASH = "/";

    private static final String TAG = "GlideUtil";

    // 图片显示的形式
    public static final int MODE_NORMAL = 1;
    public static final int MODE_CIRCLE = 2;

    private GlideUtil() {
    }

    private static class GlideHolder {
        private static GlideUtil instance = new GlideUtil();
    }

    /**
     * 使用一个静态内部类创建单例
     *
     * @return
     */
    public static GlideUtil getInstance() {
        return GlideHolder.instance;
    }

    /**
     * 加载网络图片
     *
     * @param context
     * @param url
     * @param imageView
     */
    public void displayImage(Context context, String url, ImageView imageView) {
        Glide.with(context)
                .load(url)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .fitCenter()
                .into(imageView);
    }

    /**
     * 加载网络图片并设置大小
     *
     * @param context
     * @param url
     * @param imageView
     * @param width
     * @param height
     */
    public void displayImage(Context context, String url, ImageView imageView, int width, int height) {
        Glide.with(context)
                .load(url)
                .override(width, height)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .fitCenter()
                .into(imageView);
    }

    /**
     * 加载本地图片
     *
     * @param context
     * @param file
     * @param imageView
     */
    public void displayImage(Context context, File file, ImageView imageView) {
        Glide.with(context)
                .load(file)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .fitCenter()
                .into(imageView);
    }

    /**
     * 加载本地图片
     *
     * @param context
     * @param file
     * @param imageView
     */
    public void displayImage(Context context, File file, ImageView imageView, int width, int height) {
        Glide.with(context)
                .load(file)
                .override(width, height)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .fitCenter()
                .into(imageView);
    }

    /**
     * 加载drawable图片
     *
     * @param context
     * @param resId
     * @param imageView
     */
    public void displayImage(Context context, int resId, ImageView imageView) {
        Glide.with(context)
                .load(resourceIdToUri(context, resId))
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .fitCenter()
                .into(imageView);
    }

    /**
     * 加载网络图片并显示为圆形
     *
     * @param context
     * @param url
     * @param imageView
     */
    public void displayCircleImage(Context context, String url, ImageView imageView) {
        Glide.with(context)
                .load(url)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .bitmapTransform(new GlideCircleTransform(context))
                .crossFade()
                .into(imageView);
    }

    /**
     * 加载本地图片并显示为圆形图片
     *
     * @param context
     * @param file
     * @param imageView
     */
    public void displayCircleImage(Context context, File file, ImageView imageView) {
        Glide.with(context)
                .load(file)
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .crossFade()
                .bitmapTransform(new GlideCircleTransform(context))
                .into(imageView);
    }


    /**
     * 加载drawable图片并显示为圆形图片
     *
     * @param context
     * @param resId
     * @param imageView
     */
    public void displayCircleImage(Context context, int resId, ImageView imageView) {
        Glide.with(context)
                .load(resourceIdToUri(context, resId))
                .placeholder(R.mipmap.bg_empty)
                .error(R.drawable.ic_dashboard)
                .bitmapTransform(new GlideCircleTransform(context))
                .crossFade()
                .into(imageView);
    }

    //将资源ID转为Uri
    public Uri resourceIdToUri(Context context, int resourceId) {
        return Uri.parse(ANDROID_RESOURCE + context.getPackageName() + FOREWARD_SLASH + resourceId);
    }

    public Bitmap getBitmap(Context context, String url)throws Exception {
        return Glide.with(context)
                .load(url)
                .asBitmap()
                .into(-1, -1)
                .get();
    }
}
