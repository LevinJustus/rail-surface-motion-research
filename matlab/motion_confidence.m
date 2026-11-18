function c=motion_confidence(p)
p=double(p);assert(all(isfinite(p),'all')&&all(p>=0 & p<=1,'all'));
h=zeros(size(p));a=p>0 & p<1;
h(a)=-p(a).*log2(p(a))-(1-p(a)).*log2(1-p(a));c=1-h;
end
