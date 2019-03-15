package com.example.cai.gankcamp.network;

import android.os.Handler;
import android.os.Looper;

import com.google.gson.Gson;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Created by： cai on 2016/5/8 17:41
 * Email: 429546420@qq.com
 */
public class OkHttpManager {

    private volatile static OkHttpManager instance = null;
    private OkHttpClient mOkHttpClient;
    private Handler mHandler;
    private Gson mGson;

    private OkHttpManager() {
        mOkHttpClient = new OkHttpClient();

        mHandler = new Handler(Looper.getMainLooper());
        mGson = new Gson();
    }

    public static OkHttpManager getInstance() {
        if (null == instance) {
            synchronized (OkHttpManager.class) {
                if (null == instance) {
                    instance = new OkHttpManager();
                }
            }
        }
        return instance;
    }

    //异步的get请求
    private void _getAsyn(String url, final ResultCallBack resultCallBack, Object tag) {
        Request request = new Request.Builder()
                .url(url)
                .tag(tag)
                .build();
        deliveryResult(request, resultCallBack);
    }

    private void deliveryResult(final Request request, final ResultCallBack resultCallBack) {
        mOkHttpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                sendFailedResult(call.request(), e, resultCallBack);
            }

            @Override
            public void onResponse(Call call, Response response) {
                try {
                    final String result = response.body().string();
                    sendSuccessResult(result, resultCallBack);
                    //                    if (resultCallBack.mType == String.class) {
                    //                        sendSuccessResult(result, resultCallBack);
                    //                    } else {
                    //                        Object o = mGson.fromJson(result, resultCallBack.mType);
                    //                        sendSuccessResult(o, resultCallBack);
                    //                    }
                } catch (IOException e) {
                    sendFailedResult(response.request(), e, resultCallBack);
                } catch (com.google.gson.JsonParseException e) {
                    // JSON 解析错误
                    sendFailedResult(response.request(), e, resultCallBack);
                }
            }
        });
    }

    private void sendFailedResult(final Request request, final Exception e, final ResultCallBack resultCallBack) {
        mHandler.post(new Runnable() {
            @Override
            public void run() {
                if (resultCallBack != null) {
                    resultCallBack.onError(request, e);
                }
            }
        });
    }

    private void sendSuccessResult(final Object object, final ResultCallBack resultCallBack) {
        mHandler.post(new Runnable() {
            @Override
            public void run() {
                if (resultCallBack != null) {
                    resultCallBack.onResponse(object);
                }
            }
        });
    }

    private void _cancelRequest(Object tag) {
        for (Call call : mOkHttpClient.dispatcher().queuedCalls()) {
            if (call.request().tag().equals(tag)) {
                call.cancel();
            }
        }

        for (Call call : mOkHttpClient.dispatcher().runningCalls()) {
            if (call.request().tag().equals(tag)) {
                call.cancel();
            }
        }
    }

    /*public methods*/
    public static void getAsyn(String url, final ResultCallBack resultCallBack, Object tag) {
        getInstance()._getAsyn(url, resultCallBack, tag);
    }

    public static void cancelRequest(Object tag) {
        getInstance()._cancelRequest(tag);
    }

}
