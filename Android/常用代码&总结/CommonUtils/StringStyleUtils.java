package com.example.cai.gankcamp.util;

import android.content.Context;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.TextAppearanceSpan;

import com.example.cai.gankcamp.bean.Gank;

/**
 * Created by cai on 2016/3/23.
 */
public class StringStyleUtils {

    public static SpannableString format(Context context, String text, int style) {
        SpannableString spannableString = new SpannableString(text);
        spannableString.setSpan(new TextAppearanceSpan(context, style), 0, text.length(), 0);
        return spannableString;
    }

    public static CharSequence getGankInfoSequence(Context context, Gank mGank) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(mGank.getDesc());
        return spannableStringBuilder.subSequence(0, spannableStringBuilder.length());
    }
}
