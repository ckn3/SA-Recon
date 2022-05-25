function pca_img=pca_HSI(recon_HSI,variance)
p=0;
[coeff,score,latent,tsquared,explained] = pca(recon_HSI);
 for i=1:size(recon_HSI,2)
    if sum(explained(1:i))/sum(explained)>variance
        p=i;
        break
    end
end
pca_img = score(:,1:p);
