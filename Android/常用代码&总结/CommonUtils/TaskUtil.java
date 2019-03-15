package com.example.cai.gankcamp.util;

import android.os.AsyncTask;
import android.os.Build;

/**
 * Created by： cai
 * Date：2016/4/18.
 */
public class TaskUtil {

    @SafeVarargs public static <Params, Progress, Result> void executeAsyncTask(
            AsyncTask<Params, Progress, Result> task, Params... params) {
        if (Build.VERSION.SDK_INT >= 11) {
            task.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR, params);
        }
        else {
            task.execute(params);
        }
    }
}
