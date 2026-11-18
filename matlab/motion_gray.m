function g=motion_gray(rgb)
assert(isa(rgb,'uint8') && isequal(size(rgb),[256 256 3]));
g=.2989*double(rgb(:,:,1))/255+.5870*double(rgb(:,:,2))/255+.1140*double(rgb(:,:,3))/255;
end
