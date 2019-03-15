package com.example.cai.gankcamp.network.volley;

import com.android.volley.NetworkResponse;
import com.android.volley.ParseError;
import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.toolbox.HttpHeaderParser;
import com.example.cai.gankcamp.data.API;
import com.example.cai.gankcamp.util.LogUtil;
import com.google.gson.Gson;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.UnsupportedEncodingException;
import java.lang.reflect.Type;

/**
 * Created by cai on 2016/3/30.
 */
public class GsonRequest<T> extends Request<T> {

    private final Response.Listener<T> mListener;
    private Gson mGson;
    private Class<T> mClass;
    private Type mType;

    public GsonRequest(int method, String url, Class<T> clazz, Type type, Response.Listener<T> listener,
                       Response.ErrorListener errorListener) {
        super(method, url, errorListener);
        mGson = new Gson();
        mClass = clazz;
        mType = type;
        mListener = listener;
    }

    public GsonRequest(String url, Class<T> clazz, Type type,Response.Listener<T> listener,
                       Response.ErrorListener errorListener) {
        this(Method.GET, url, clazz, type, listener, errorListener);
    }

    @Override
    protected Response<T> parseNetworkResponse(NetworkResponse networkResponse) {
        try {
            String jsonString = new String(networkResponse.data, HttpHeaderParser.parseCharset(networkResponse.headers));
            JSONObject jsonObject = null;
            String results = "";
            try {
                jsonObject = new JSONObject(jsonString);
                results = jsonObject.getString(API.API_RESULTS);
            } catch (JSONException e) {
                LogUtil.e("GsonRequest", "parse json error: " + e.getMessage());
            }
            T resultsT = mGson.fromJson(results, mType);
            return Response.success(resultsT, HttpHeaderParser.parseCacheHeaders(networkResponse));
        } catch (UnsupportedEncodingException e) {
            return Response.error(new ParseError(e));
        }
    }

    @Override
    protected void deliverResponse(T response) {
        mListener.onResponse(response);
    }
}
