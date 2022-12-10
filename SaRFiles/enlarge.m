function train_enlarge=enlarge(train_original,img_original,label_original)


train_add=[];
global scale_sp;
scale_sp = [1 2 3 5 7 9];

row_max = size(label_original,1);
col_max = size(label_original,2);

pca1 = PCA_3d(img_original,1);

gm = 385;
len = shape_adaptive(pca1,gm);

len = reshape(len,row_max*col_max,[])';
len = scale_sp(len);

for i=1:size(train_original,2)
    row = mod(train_original(1,i),row_max);
    if row == 0
        row = row_max;
    end
    col = ceil(train_original(1,i)/row_max);
    lens = len(:,train_original(1,i));
   
    pixs_xy=PtsSaR(lens,row,col);
    in=sub2ind([row_max,col_max],pixs_xy(:,1),pixs_xy(:,2))';
    label=train_original(2,i)*ones(1,length(in));
    add=cat(1,in,label);
    train_add=[train_add add];

end
train_adds=[train_original train_add];
train_SR=train_adds';
train_SR=unique(train_SR,'rows');
train_enlarge=train_SR';
