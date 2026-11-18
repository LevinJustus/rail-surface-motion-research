function [warped,valid,displacement]=warp_motion_history(history,shifts)
% shifts are chronological forward [dx dy] edges from history to current.
assert(isequal(size(history),[256 256]) && size(shifts,2)==2 && all(isfinite(shifts),'all'));
[x,y]=meshgrid(1:256,1:256);qx=x;qy=y;valid=true(256);
for j=size(shifts,1):-1:1
    qx=qx-shifts(j,1);qy=qy-shifts(j,2);
    valid=valid & qx>=1 & qx<=256 & qy>=1 & qy<=256;
end
displacement=sum(shifts,1);
warped=interp2(double(history),qx,qy,'linear',0);warped(~valid)=0;
assert(all(isfinite(warped),'all'));
end
