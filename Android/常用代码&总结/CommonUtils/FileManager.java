package com.example.cai.gankcamp.util;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import com.example.cai.gankcamp.data.Constant;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Created by： cai
 * Date：2016/4/18.
 */
public class FileManager {
    private static final String TAG = "FileManager";

    public static FileManager instance;
    Context ctx;

    private File currentDir;
    public static final String imageDirName = "gankCamp_image";

    public static FileManager getInstance(Context context) {
        if (instance == null) {
            instance = new FileManager(context);
        }
        return instance;
    }

    private FileManager(Context context) {
        ctx = context;
    }

    /**
     * 创建文件夹及文件
     *
     * @param dirName
     * @param filename
     */
    private File createDirFolder(String dirName, String filename) {
        if (isSdEmpty()) {
            ToastUtil.showShort(Constant.ERROR_SD_EMPTY);
            return null;
        }
        File appDir = new File(getSDPath(), dirName);
        if (!appDir.exists()) {
            boolean is = appDir.mkdir();
            if (is) {
                LogUtil.d(TAG, "create success");
            } else {
                LogUtil.d(TAG, "create failed");
            }
        }
        File file = new File(appDir, filename);

        currentDir = appDir;
        return file;
    }

    public boolean saveBitmap(Bitmap bitmap, String dirName, String filename) {
        // 创建文件
        File imageFile = createDirFolder(dirName, filename);
        // 保存图片
        try {
            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        // 其次把文件插入到系统图库
        try {
            MediaStore.Images.Media.insertImage(ctx.getContentResolver(),
                    imageFile.getAbsolutePath(), filename, null);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // 最后通知图库更新
        Intent scannerIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE,
                Uri.parse("file://" + imageFile.getAbsolutePath()));
        ctx.sendBroadcast(scannerIntent);

        return true;
    }

    public File getCurrentDir() {
        return currentDir;
    }

    public File getImageFile(String filename) {
        return createDirFolder(imageDirName, filename);
    }


    // 判断sd卡是否为空
    private boolean isSdEmpty() {
        return !Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED);
    }

    public String getSDPath() {
        return Environment.getExternalStorageDirectory().toString();
    }

}
