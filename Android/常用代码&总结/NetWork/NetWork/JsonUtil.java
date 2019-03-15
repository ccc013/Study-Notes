package com.example.cai.gankcamp.network;

import com.example.cai.gankcamp.bean.Gank;
import com.example.cai.gankcamp.data.API;
import com.example.cai.gankcamp.model.onLoadDataListener;
import com.example.cai.gankcamp.util.LogUtil;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cai on 2016/3/30.
 */
public class JsonUtil {

    private static final String TAG = "JsonUtil";

    public static ArrayList<Gank> parseJSONWithJSONObject(String jsonData) {
        ArrayList<Gank> mGanks = new ArrayList<Gank>();
        try {
            // 将返回值变成一个JSON对象
            JSONObject results = new JSONObject(jsonData);
            JSONArray jsonArray = results.getJSONArray(API.API_RESULTS);
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                Gank gank = new Gank();
                gank.setDesc(jsonObject.getString("desc"));
                gank.setType(jsonObject.getString("type"));
                gank.setUrl(jsonObject.getString("url"));
                gank.setWho(jsonObject.getString("who"));
                gank.setPublishedAt(jsonObject.getString("publishedAt"));
                mGanks.add(gank);
            }
        } catch (Exception e) {
            LogUtil.e(TAG, "parseJSON error: " + e.getMessage());
        }

        return mGanks;
    }

    public static ArrayList<Gank> parseJsonWithGSON(String jsonData) {
        ArrayList<Gank> mGanks = new ArrayList<Gank>();
        try {
            Gson gson = new Gson();
            Type type = new TypeToken<List<Gank>>() {
            }.getType();
            mGanks = gson.fromJson(jsonData, type);
        } catch (Exception e) {
            LogUtil.e(TAG, "parseJsonWithGson error: " + e.getMessage());
        }

        return mGanks;
    }

    public static void handleResponse(String response, String url, onLoadDataListener listener) {
        try {
            JSONObject resultsJson = new JSONObject(response);
            //check whether the error key exists,
            if (resultsJson.isNull(API.API_ERROR)) {
                listener.onError("error key not exists");
                return;
            }
            //return the error key, true is failure, false is successful
            if (resultsJson.getBoolean(API.API_ERROR)) {
                listener.onError("request failure");
                return;
            }

            if (resultsJson.isNull(API.API_RESULTS)) {
                listener.onError("results key not exists!");
                return;
            }
            // return the data
            String results = resultsJson.getString(API.API_RESULTS);

            listener.onSuccess(results);

        } catch (JSONException e) {
            listener.onError(e.getLocalizedMessage());
        }
    }
}
