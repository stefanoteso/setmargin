k = 3; % set cardinality
m = 5; % number of attributes
lambda = 0.1; % weight/penalty of slack variables
beta = 0.1; % weight/penalty L1 norm
gamma = 0.1; % 

bigC = 10000; % an arbitrary big constant

% upper bound of W
w_UP = 1;

% preference data (cell array of structs representint pairwise comparisons)
pairwisePref1 = struct('plus',[1,1,0,0,0],'minus',[0,0,1,0,0]);
pairwisePref2 = struct('plus',[0,0,0,0,1],'minus',[0,1,0,1,0]);
pairwisePref3 = struct('plus',[1,0,0,0,1],'minus',[0,1,0,0,0]);

prefs={pairwisePref1, pairwisePref2, pairwisePref3}; 
n_prefs = numel(prefs);

% indices to decision variables
% W
W_I_start = zeros(1,k);
W_I_end = zeros(1,k);
W_I = zeros(k,m);
iter_index = 0;
for i=1:k
    iter_index = iter_index + 1;
    W_I_start(i) = iter_index;
    W_I(i,:) = iter_index : (iter_index + (m-1));
    iter_index = iter_index + (m-1);
    W_I_end(i) = iter_index;
end
% X
X_I_start = zeros(1,k);
X_I_end = zeros(1,k);
X_I = zeros(k,m);
for i=1:k
    iter_index = iter_index + 1;
    X_I_start(i) = iter_index;
    X_I(i,:) = iter_index : (iter_index + (m-1));
    iter_index = iter_index + (m-1);
    X_I_end(i) = iter_index;
end

% epsilon
if n_prefs > 0
    epsilon_I = zeros(k,n_prefs);
    epsilon_I_start = iter_index + 1;
    for i=1:k
        for z=1:n_prefs
            iter_index = iter_index + 1;
            epsilon_I(i,z) = iter_index;
        end
    end
    epsilon_I_end = iter_index;
end

% A
A_I = zeros(k,k,m);
for i=1:k
    for j=1:k
        for z=1:m
            iter_index = iter_index + 1;
            A_I(i,j,z) = iter_index;
        end
    end
end

% M
iter_index = iter_index + 1;
M_I = iter_index;

nDecVar = iter_index;

% variable types
varType = repmat('C',1,nDecVar);
varType(X_I_start(1):X_I_end(k))='B';

% objective function
fobj = zeros(1,nDecVar);
fobj(M_I)=1;
fobj(epsilon_I_start:epsilon_I_end)=-lambda;
fobj(W_I_start(1):W_I_end(end))=-beta;
for i=1:k
    fobj(A_I(i,i,:))=gamma;
end

% constraints: preference data
if n_prefs > 0 
    c_prefs = zeros(n_prefs*k,nDecVar);
    c_prefs_rhs = zeros(n_prefs*k,1);
    c_prefs_sense = repmat('>',1,n_prefs*k);
    iter_c=0;
    for i=1:k
        for l=1:n_prefs
            iter_c = iter_c + 1;
            prefConstraint = zeros(1,nDecVar);
            y_plus = prefs{l}.plus;
            y_minus = prefs{l}.minus;
            prefConstraint(W_I_start(i):W_I_end(i))=y_plus-y_minus; 
            prefConstraint(M_I)=-1;
            prefConstraint(epsilon_I(k,l))=1;
            c_prefs(iter_c,:)=prefConstraint;
        end
    end
else
    c_prefs = []; c_prefs_rhs = []; c_prefs_sense = '';
end

% constraints: \sum A^{i,i}_z - A^{i,j}_z >= M
c_sum_A_minus_A = zeros(k*(k-1),nDecVar);
c_sum_A_minus_A_rhs = zeros(k*(k-1),1);
c_sum_A_minus_A_sense = repmat('>',1,k*(k-1));
iter_c = 0;
for i=1:k
    for j=1:k
        if i==j
            continue
        end
        iter_c = iter_c + 1;
        thisConstraint = zeros(1,nDecVar);
        thisConstraint(A_I(i,i,1):A_I(i,i,m))=1;
        thisConstraint(A_I(i,j,1):A_I(i,j,m))=-1;
        thisConstraint(M_I)=-1;
        % TODO: epsilon_prime
        c_sum_A_minus_A(iter_c,:)=sparse(thisConstraint);
    end
end

% constraints: A^{i,i}_z < ub(w) x^i_z
c_A_le_X = zeros(k*m,nDecVar);
c_A_le_X_rhs = zeros(k*m,1);
c_A_le_X_sense = repmat('<',1,k*m);
iter_c = 0;
for i=1:k
    for z=1:m
        iter_c = iter_c + 1;
        thisConstraint = zeros(1,nDecVar);
        thisConstraint(A_I(i,i,z))=1;
        thisConstraint(X_I(i,z))=-w_UP;
        c_A_le_X(iter_c,:)=thisConstraint;
    end
end

% constraints: A^{i,i}_z < w^i_z
c_A_le_W = zeros(k*m,nDecVar);
c_A_le_W_rhs = zeros(k*m,1);
c_A_le_W_sense = repmat('<',1,k*m);
iter_c = 0;
for i=1:k
    for z=1:m
        iter_c = iter_c + 1;
        thisConstraint = zeros(1,nDecVar);
        thisConstraint(A_I(i,i,z))=1;
        thisConstraint(W_I(i,z))=-1;
        c_A_le_W(iter_c,:)=thisConstraint;
    end
end

% constraints: A^{i,j}_z > w^i_z - C (1 - x^j_z)
% equiv formulation: A^{i,j}_z - w^i_z - C x^j_z >  - C 
c_A_ge_W_minus_C = zeros(k*(k-1)*m,nDecVar);
c_A_ge_W_minus_C_rhs = - bigC * ones(k*(k-1)*m,1);
c_A_ge_W_minus_C_sense = repmat('>',1,k*(k-1)*m);
iter_c = 0;
for i=1:k
    for j=1:k
        if i==j
            continue
        end
        for z=1:m
            iter_c = iter_c + 1;
            thisConstraint = zeros(1,nDecVar);
            thisConstraint(A_I(i,j,z))=1;
            thisConstraint(W_I(i,z))=-1;
            thisConstraint(X_I(j,z))=-bigC;
            c_A_ge_W_minus_C(iter_c,:)=thisConstraint;
        end
    end
end

% constraints: A^{i,j}_z > 0
% (expressed as lower bound)

% pack data
clear model;
model.obj=fobj;
model.vtype=varType;
model.A=sparse([c_prefs; c_sum_A_minus_A; c_A_le_X; c_A_le_W; c_A_ge_W_minus_C]);
model.modelsense = 'max';
model.rhs=[c_prefs_rhs; c_sum_A_minus_A_rhs; c_A_le_X_rhs; c_A_le_W_rhs; c_A_ge_W_minus_C_rhs];
model.sense=[c_prefs_sense, c_sum_A_minus_A_sense, c_A_le_X_sense, c_A_le_W_sense, c_A_ge_W_minus_C_sense];
model.ub = repmat(bigC,1,nDecVar);
model.ub(W_I_start(1):W_I_end(k)) = w_UP;


% parameters
clear params;
params.outputflag = 0;

% invoke solver
result = gurobi(model, params);

%disp(result);

W = zeros(k,m);
X = zeros(k,m);
epsilon = zeros(k,n_prefs);
A = zeros(k,k,m);
for i=1:k
    W(i,:)=result.x(W_I(i,:));
    X(i,:)=result.x(X_I(i,:));
    if n_prefs > 0; epsilon(i,:)=result.x(epsilon_I(i,1):epsilon_I(i,end)); end
    for j=1:k
        A(i,j,:)=result.x(A_I(i,j,:));
    end
end
M = result.x(M_I);

% display results
fprintf(' Note: results are to be read row-by-row \n');
disp('X');
disp(X);
disp('W');
disp(W);
disp('M');
disp(M);