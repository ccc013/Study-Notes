package com.example.cai.gankcamp.util;

import android.content.Context;
import android.widget.Toast;

import com.example.cai.gankcamp.GankApp;

/**
 * Created by cai on 2016/3/24.
 */
public class ToastUtil {

    public static void show(Context context, String content, int duration) {
        Toast.makeText(context, content, duration).show();
    }

    public static void show(Context context, int ResId, int duration) {
        Toast.makeText(context, ResId, duration).show();
    }

    public static void showShort(String content) {
        Toast.makeText(GankApp.mContext, content, Toast.LENGTH_SHORT).show();
    }

    public static void showShort(int ResId) {
        ToastUtil.show(GankApp.mContext, ResId, Toast.LENGTH_SHORT);
    }

    public static void showLong(String content) {
        ToastUtil.show(GankApp.mContext, content, Toast.LENGTH_LONG);
    }

    public static void showLong(int ResId) {
        ToastUtil.show(GankApp.mContext, ResId, Toast.LENGTH_LONG);
    }
}
