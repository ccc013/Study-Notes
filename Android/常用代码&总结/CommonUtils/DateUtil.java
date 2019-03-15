package com.example.cai.gankcamp.util;

import android.text.TextUtils;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

/**
 * Created by cai on 2016/3/24.
 */
public class DateUtil {


    public static String dateToString(Date date) {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy/MM/dd");
        return simpleDateFormat.format(date);
    }

    public static String dateToString(Date date, String format) {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(format);
        return simpleDateFormat.format(date);
    }

    public static Date stringToDate(String dateString) {
        Date date = null;
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");

        try {
            date = format.parse(dateString);
        } catch (Exception e) {
            LogUtil.e("DateUtil", "stringToDate error: " + e.getMessage());
        }
        return date;
    }

    /**
     *  change the original date to another format's date
     * @param old_date
     * @param format
     * @return
     */
    public static String formatToDate(String old_date, String format) {
        Date new_date = stringToDate(old_date);
        if (!TextUtils.isEmpty(format)) {
            return dateToString(new_date, format);
        } else {
            return dateToString(new_date);
        }
    }

    public static String addDate(Date date, int add) {
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        calendar.add(Calendar.DATE, add);
        return dateToString(calendar.getTime());
    }

    public static Date getLastdayDate(Date date) {
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        calendar.add(Calendar.DATE, -1);
        return calendar.getTime();
    }

    public static Date getNextdayDate(Date date) {
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        calendar.add(Calendar.DATE, 1);
        return calendar.getTime();
    }
}
