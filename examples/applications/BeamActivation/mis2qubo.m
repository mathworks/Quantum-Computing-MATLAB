function qprob = mis2qubo(first, w, options)
% MIS2QUBO  Convert a weighted maximum independent set problem to QUBO.
%
%   QPROB = MIS2QUBO(G) converts the weighted maximum independent set
%   problem on graph G to a QUBO object. Node weights are read from
%   G.Nodes.Weight if present; otherwise unit weights are used.
%
%   QPROB = MIS2QUBO(A) uses adjacency matrix A (n-by-n) with unit node
%   weights.
%
%   QPROB = MIS2QUBO(A, W) uses adjacency matrix A and node weight vector
%   W of length n.
%
%   QPROB = MIS2QUBO(___, Penalty=P) sets the penalty coefficient applied
%   to each conflict edge (default: 1). Set P > max(W) to ensure that no
%   reward justifies a constraint violation.
%
%   Output:
%     QPROB  - qubo object encoding:
%                minimize  -sum_i w(i)*x(i) + P*sum_{(i,j) in E} x(i)*x(j)
%
%   The optimal x minimising QPROB energy corresponds to the maximum
%   weight independent set of G (with the sign convention that selecting
%   node i contributes -w(i) to the energy).
%
%   Example — 3-node path graph, unit weights:
%     G     = graph([1 2],[2 3]);
%     qprob = mis2qubo(G, Penalty=5);
%     sol   = solve(qprob, Algorithm="tabu");
%     % Optimal MIS is nodes {1,3}; sol.BestX = [1 0 1]
%
%   See also KNAPSACK2QUBO, TSP2QUBO, QUBO, ADDINEQUALITYPENALTY.

arguments
    first
    w     {mustBeNumeric} = double.empty(0,1)
    options.Penalty (1,1) double {mustBePositive} = 1
end

penalty = options.Penalty;

% --- extract adjacency matrix and node weights ----------------------------
if isa(first, 'graph')
    if ~isempty(w)
        error('mis2qubo:invalidInput', ...
            'Specify node weights via G.Nodes.Weight when passing a graph object, not as a second argument.');
    end
    G = first;
    n = numnodes(G);
    if ismember('Weight', G.Nodes.Properties.VariableNames)
        nodeWeights = G.Nodes.Weight(:);
    else
        nodeWeights = ones(n, 1);
    end
    A = adjacency(G);

elseif isnumeric(first) && ismatrix(first)
    [nr, nc] = size(first);
    if nr ~= nc
        error('mis2qubo:invalidInput', 'Adjacency matrix must be square.');
    end
    n = nr;
    % Symmetrize user-supplied adjacency (handles upper-tri, lower-tri, or full input)
    A = logical(first) | logical(first');
    if isempty(w)
        nodeWeights = ones(n, 1);
    else
        nodeWeights = w(:);
        if numel(nodeWeights) ~= n
            error('mis2qubo:invalidInput', ...
                'Length of W (%d) must match the number of nodes (%d).', ...
                numel(nodeWeights), n);
        end
    end

else
    error('mis2qubo:invalidInput', ...
        'First argument must be a graph object or a numeric adjacency matrix.');
end

% --- build symmetric Q matrix ---------------------------------------------
Q = diag(-nodeWeights);

% A is already symmetric (adjacency(G) for graph input; symmetrized above
% for matrix input). Iterate upper triangle to avoid double-counting edges.
[row, col] = find(triu(A, 1));
for k = 1:numel(row)
    i = row(k); j = col(k);
    Q(i,j) = Q(i,j) + penalty/2;
    Q(j,i) = Q(j,i) + penalty/2;
end

qprob = qubo(Q);
end
