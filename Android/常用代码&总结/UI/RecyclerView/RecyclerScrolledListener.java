package com.example.cai.gankcamp;

import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.View;

/**
 * Created by： cai on 2016/5/15 13:15
 * Email: 429546420@qq.com
 */
public abstract class RecyclerScrolledListener extends RecyclerView.OnScrollListener {
    private static final int HIDE_THRESHOLD = 20;
    private int scrollDistance = 20;
    private boolean controlVisible = true;

    @Override
    public void onScrolled(RecyclerView recyclerView, int dx, int dy) {
        super.onScrolled(recyclerView, dx, dy);
        int firstVisbleItem = ((LinearLayoutManager) recyclerView.getLayoutManager()).findFirstVisibleItemPosition();
        if (firstVisbleItem == 0) {
            if (!controlVisible) {
                onShow();
                controlVisible = true;
            }
        } else {
            if (scrollDistance > HIDE_THRESHOLD && controlVisible) {
                onHide();
                controlVisible = false;
                scrollDistance = 0;
            } else if (scrollDistance < -HIDE_THRESHOLD && !controlVisible) {
                onShow();
                controlVisible = true;
                scrollDistance = 0;
            }
        }
        if ((controlVisible && dy > 0) || (!controlVisible && dy < 0)) {
            scrollDistance += dy;
        }
        View lastChildView = recyclerView.getLayoutManager().getChildAt(recyclerView.getLayoutManager().getChildCount() - 1);
        int lastPosition = recyclerView.getLayoutManager().getPosition(lastChildView);
        if ( lastPosition == recyclerView.getLayoutManager().getItemCount() - 4) {
            loadMore();
        }
    }

    public abstract void onHide();

    public abstract void onShow();

    public abstract void loadMore();
}
