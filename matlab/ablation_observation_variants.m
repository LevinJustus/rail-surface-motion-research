function variants=ablation_observation_variants(frames)
% Absolute-observation perturbations only; preserve original missing flags.
variants=cell(7,1);bias=[-6 -2 2 6];
for j=1:7
    f=frames;
    for k=1:numel(f)
        if j==7 || ~f(k).odometry_available
            f(k).odometry_available=false;f(k).simulated_odometry_px=NaN;
        elseif j<=4,f(k).simulated_odometry_px=f(k).simulated_odometry_px+bias(j)*(k-1);
        else
            amplitude=1;if j==6,amplitude=3;end
            f(k).simulated_odometry_px=f(k).simulated_odometry_px+amplitude*(-1)^(k-1);
        end
    end
    variants{j}=f;
end
end
