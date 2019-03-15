package com.example.cai.gankcamp.network.http;

import android.os.Handler;

import com.example.cai.gankcamp.model.onLoadDataListener;
import com.example.cai.gankcamp.network.volley.VolleyManager;

/**
 * Created by cai on 2016/3/20.
 */
public class AsynHttpUtil {

    public static void get(final String url, final onLoadDataListener listener) {
        final Handler handler = new Handler();
        new Thread(new Runnable() {
            @Override
            public void run() {
                // final String response = HttpManager.sendHttpRequest(url);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        //JsonUtil.handleResponse(response, url, false, listener);
                        VolleyManager.getInstance().get(url, listener);
                    }
                });
            }
        }).start();
    }


}
