package com.example.cai.gankcamp.network.volley;

import android.content.Context;

import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.ImageLoader;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.cai.gankcamp.data.Constant;
import com.example.cai.gankcamp.model.onLoadDataListener;
import com.example.cai.gankcamp.network.JsonUtil;
import com.example.cai.gankcamp.util.LogUtil;

import org.json.JSONObject;

/**
 * Created by cai on 2016/3/30.
 */
public class VolleyManager {

    private volatile static VolleyManager instance = null;

    private RequestQueue mRequestQueue = null;

    private ImageLoader mImageLoader = null;

    private BitmapCache mBitmapCache;


    private VolleyManager() {

    }


    public static VolleyManager getInstance() {
        if (null == instance) {
            synchronized (VolleyManager.class) {
                if (null == instance) {
                    instance = new VolleyManager();
                }
            }
        }
        return instance;
    }

    public void init(Context context) {
        if (this.mRequestQueue == null) {
            this.mRequestQueue = Volley.newRequestQueue(context);
        }

        mImageLoader = new ImageLoader(VolleyManager.getInstance().getRequestQueue(), mBitmapCache);
    }

    public RequestQueue getRequestQueue() {
        return this.mRequestQueue;
    }

    public void get(final String url, final onLoadDataListener listener) {
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(url, null, new Response.Listener<JSONObject>() {
            @Override
            public void onResponse(JSONObject jsonObject) {
                LogUtil.d("volleyManager", "onResponse");
                JsonUtil.handleResponse(jsonObject.toString(), url, listener);
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError volleyError) {
                listener.onError(Constant.NET_getDataFailed);
            }
        });
        jsonObjectRequest.setTag(Constant.REQUEST_TAG);
        this.mRequestQueue.add(jsonObjectRequest);
    }

    // cancel all request whose tag is Constant.REQUEST_TAG
    public void cancelAll() {
        this.mRequestQueue.cancelAll(Constant.REQUEST_TAG);
    }
}
