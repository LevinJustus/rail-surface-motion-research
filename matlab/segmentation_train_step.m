function [newNet,newAvg,newAvgSq,stats] = segmentation_train_step(net,avg,avgSq,iteration,X,T,V,cfg,logitsName)
%SEGMENTATION_TRAIN_STEP One finite, globally clipped Adam update on training data.
% Same mathematical update as the verified Step08 loop; no data selection.
[loss,gradients,state,bce,dice]=dlfeval(@lossAndGradient,net, ...
    dlarray(X,'SSCB'),T,V,cfg.loss,logitsName);
stats.total_loss=double(extractdata(loss));
stats.BCE_component=double(extractdata(bce));stats.Dice_component=double(extractdata(dice));
assert(all(isfinite([stats.total_loss stats.BCE_component stats.Dice_component])),'Nonfinite loss');
sq=0;nonzero=0;
for k=1:height(gradients)
    g=extractdata(gradients.Value{k});assert(all(isfinite(g),'all'),'Nonfinite gradient');
    sq=sq+sum(double(g).^2,'all');nonzero=nonzero+nnz(g);
end
norm=sqrt(sq);assert(isfinite(norm) && norm>0 && nonzero>0,'No effective gradient');
factor=min(1,cfg.optimizer.max_gradient_norm/norm);
for k=1:height(gradients),gradients.Value{k}=gradients.Value{k}*single(factor);end
[newNet,newAvg,newAvgSq]=adamupdate(net,gradients,avg,avgSq,iteration, ...
    cfg.optimizer.learning_rate,cfg.optimizer.beta1,cfg.optimizer.beta2,cfg.optimizer.epsilon);
newNet.State=state;deltaSq=0;
for k=1:height(net.Learnables)
    old=extractdata(net.Learnables.Value{k});new=extractdata(newNet.Learnables.Value{k});
    assert(all(isfinite(new),'all'),'Nonfinite new parameter');
    deltaSq=deltaSq+sum((double(new)-double(old)).^2,'all');
end
assert(deltaSq>0,'Parameters did not change');
stats.unclipped_global_gradient_norm=norm;stats.clip_factor=factor;
stats.gradient_nonzero_elements=nonzero;stats.parameter_delta_L2=sqrt(deltaSq);
end

function [loss,gradients,state,bce,dice]=lossAndGradient(net,X,T,V,cfg,logitsName)
[Z,state]=forward(net,X,Outputs=logitsName);
[loss,bce,dice]=masked_segmentation_loss(Z,T,V,cfg);
gradients=dlgradient(loss,net.Learnables);
end
