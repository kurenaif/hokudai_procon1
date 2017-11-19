#include <iostream>
#include <vector>
#include <unordered_map>
#include <ciso646>
#include <algorithm>
#include <cmath>
#include <climits>
#include <random>
#include <set>
#include <queue>
#include <map>


#define FOR(i,a,b) for (int i=(a);i<(b);i++)
#define RFOR(i,a,b) for (int i=(b)-1;i>=(a);i--)
#define REP(i,n) for (int i=0;i<(n);i++)
#define RREP(i,n) for (int i=(n)-1;i>=0;i--)


using namespace std;



class xor128 {
public:
	static constexpr unsigned min() { return 0u; }   // 乱数の最小値
	static constexpr unsigned max() { return UINT_MAX; } // 乱数の最大値
	unsigned operator()() { return random(); }
	xor128() {
		std::random_device rd;
		w = rd();
	}
	xor128(unsigned s) { w = s; }  // 与えられたシードで初期化
	unsigned random() {
		unsigned t;
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
	}
	unsigned random(int high) {
		return random()%high;
	}
	unsigned random(int low, int high) {
		return random()%(high-low) + low;
	}
private:
	unsigned x = 123456789u, y = 362436069u, z = 521288629u, w;
};

class Graph {
public:
	vector<vector<int> > G;
	vector<vector<int> > mat;
	vector<long long> edgeSum;
	Graph(const int nodeNum) :G(nodeNum), mat(nodeNum, vector<int>(nodeNum, 0)), edgeSum(nodeNum) {}
	void add_edge(const int from, const int to, int cost) {
		if (cost == 0) return;
		G[from].push_back(to);
		G[to].push_back(from);
		edgeSum[from] = -1;
		edgeSum[to] = -1;
		mat[from][to] = cost;
		mat[to][from] = cost;
	}
	void change_edge(const int from, const int to, int cost) {
		if (mat[from][to] != 0) G[from].push_back(to);
		if (mat[to][from] != 0) G[from].push_back(from);
		mat[from][to] = cost;
		mat[to][from] = cost;
	}
	const vector<int> & get_to(const int node) const {
		return G[node];
	}
	const long get_edge_sum(const int node){
		if(edgeSum[node] == -1){
			edgeSum[node] = 0;
			priority_queue<int> q;
			for(auto &a:mat[node]){
				q.push(a);
			}
			for(int i=0;i<10 and not q.empty();++i){
				int val = q.top(); q.pop();
				edgeSum[node] += val;
			}
		}
		return edgeSum[node];
	}
	int size() const {
		return G.size();
	}
	const int get_cost(int from, int to) const {
		return mat[from][to];
	}
};

class Check {
public:
	vector<int> checkArray;
	vector<bool> isChecked;
	Check(int node) :checkArray(), isChecked(node) {
		checkArray.reserve(node);
	}
	void check(int node) {
		checkArray.push_back(node);
		isChecked[node] = true;
	}
	bool get_is_checked(int node) const { return isChecked[node]; }
	const vector<int>& get_checked_nodes() const { return checkArray; }
	vector<int> get_not_checks() const {
		vector<int> res;
		REP(i, isChecked.size()) if (not isChecked[i]) { res.push_back(i); }
		return res;
	}
};
bool operator < (const Check& left, const Check& right){
    return left.isChecked < right.isChecked;
}
bool operator == (const Check& left, const Check& right){
	return left.isChecked == right.isChecked;
}
bool operator != (const Check& left, const Check& right){
	return not (left.isChecked == right.isChecked);
}
struct CheckHash {
	typedef std::size_t result_type;
	std::size_t operator()(const Check& key) const{
		return std::hash<vector<bool>>()(key.isChecked);
	};
};


void map_graph(Check& envCheck, int envNode, unordered_map<int, int>& phi, Check& gCheck, int gNode) {
	phi[envNode] = gNode;
	envCheck.check(envNode);
	gCheck.check(gNode);
}

struct State{
	unordered_map<int, int> phi;
	Check envCheck;
	Check gCheck;
	int score;
	State(const unordered_map<int, int>& phi, const Check& envCheck, const Check& gCheck, const int score):phi(phi), envCheck(envCheck), gCheck(gCheck), score(score){}
	State(const unordered_map<int, int>& phi, const Check& envCheck, const Check& gCheck):phi(phi), envCheck(envCheck), gCheck(gCheck), score(0){}
};

struct StateHash {
	typedef std::size_t result_type;
	std::size_t operator()(const State& key) const{
		return CheckHash()(key.gCheck) ^ CheckHash()(key.envCheck);
	};
};
bool operator < (const State& left, const State& right){
    if(left.score != right.score) return left.score < right.score;
	if(left.envCheck != right.envCheck) return left.envCheck < right.envCheck;
    return left.gCheck < right.gCheck;
}
bool operator == (const State& left, const State& right){
	return left.gCheck == right.gCheck and left.envCheck == right.envCheck;
}
unordered_map<State, vector<State>, StateHash> stateMemo;// state, size: value vector<State>
vector<State> score(const Graph& G, const Graph& envG, const State& state, int size){
	if(stateMemo.find(state) != stateMemo.end()) {
		return stateMemo[state];
	}
	const unordered_map<int, int>& phi = state.phi;
	const Check& envCheck = state.envCheck;
	const Check& gCheck = state.gCheck;
    if(phi.size() == G.size()) return {};
    vector<State> res(size, state);
	unordered_map<int, vector<int>> envNodes;
	for (auto &n : envCheck.get_checked_nodes()) for (auto &t : envG.get_to(n)) if (not envCheck.get_is_checked(t)) {
				envNodes[t].push_back(n);
			}
	unordered_map<int, pair<int, double>> envScores; // env側でのnodeの評価値 envScores[envTo] = map(gTo, score);
	for (const pair<int, vector<int> > &nodePair : envNodes) { // ↑の処理でやったenv側のノード first:to, second: froms
		unordered_map<int, double> gScores; // scores[to] = score;
		//G側でスコアの精算
		set<int> gCheckNodes; // g側でcheckしたnode
		vector<int> notChecked = gCheck.get_not_checks();
		for (const int &envCheckedNode : nodePair.second) {
			int from = phi.at(envCheckedNode);// envG -> G
			for (int to : notChecked) { // unchecked G nodes
				gScores[to] += G.get_cost(from, to);
				gCheckNodes.insert(to);
			}
		}
		//G側で最大値
		pair<int, double> best = *std::max_element( //first: to, second: score
				gScores.begin(), gScores.end(),
				[](const pair<int, int>& left, const pair<int, int>& right) {
					return left.second < right.second;
				}
		);
		envScores[nodePair.first] = best;
	}
	//env側で最大値
	pair<int, pair<int, double>> best = *std::max_element( //first: envTo, second: (first: gTo, score)
			envScores.begin(), envScores.end(),
			[](const pair<int, pair<int, int>>& left, const pair<int, pair<int, int>>& right) {
				return left.second.second < right.second.second;
			}
	);
	vector<pair<double, pair<int, int> > > bestNodes;
	for(const auto &a:envScores){
		bestNodes.push_back({a.second.second, {a.first, a.second.first}});
	}
	sort(bestNodes.begin(), bestNodes.end());
	reverse(bestNodes.begin(), bestNodes.end());
	//map graph
    REP(i,res.size()){
        State& s = res[i];
		map_graph(s.envCheck, bestNodes[i].second.first, s.phi, s.gCheck, bestNodes[i].second.second);
        s.score = bestNodes[i].first;
	}
	stateMemo[state] = res;
	return res;
}

int DFS(const Graph& G, const Graph& envG, const State& state, int width, int depth){
	if(depth == 0 or state.phi.size() == G.size()) return state.score;
	vector<State> next = score(G, envG, state, width);
	int ma = state.score;
	REP(i,next.size()){
		ma = max(DFS(G, envG, next[i], width, depth-1), ma);
	}
	return ma;
}


int main(void) {
	// fast cin
	cin.tie(0);
	ios::sync_with_stdio(false);
	xor128 random;
	// input
	int V, E; cin >> V >> E;
	Graph G(V); //0-indexed node
	Check gCheck(V);
	REP(i, E) {
		int u, v, w; cin >> u >> v >> w;
		--u; --v;
		G.add_edge(u, v, w);
	}
	cin >> V >> E;
	Graph envG(V);
	Check envCheck(V);
	REP(i, E) {
		int u, v; cin >> u >> v;
		--u; --v;
		envG.add_edge(u, v, -1);
	}

	unordered_map<int, int> phi;
	// first node
	pair<int, int> bestFirst; // first:score, second:node
	REP(from, G.size()) {
		int score = G.get_edge_sum(from);
		if (bestFirst.first < score) {
			bestFirst.first = score;
			bestFirst.second = from;
		}
	}
	int center_node = envG.size() / 2;
	if (envG.size() % 2 == 0) center_node -= sqrt(envG.size()) / 2;
	map_graph(envCheck, center_node, phi, gCheck, bestFirst.second);

    queue<State> que;
	que.push(State(phi, envCheck, gCheck, 0));

	vector<State> lasts;
    while(not que.empty()) {
		State state = que.front(); que.pop();
		if(state.phi.size() == G.size()){
			lasts.push_back(state);
            continue;
		}
        vector<State> next = score(G, envG, state, 2);
		vector<int> points(next.size());
        REP(i,next.size()){
			points[i] = DFS(G, envG, next[i], 2, 1);
		}
        int index = max_element(points.begin(), points.end()) - points.begin();
        que.push(next[index]);
	}
    State res = *max_element(lasts.begin(), lasts.end());
	for (auto &a : res.phi) {
		cout << a.second + 1 << " " << a.first + 1 << endl;
	}
}