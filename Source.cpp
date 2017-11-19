#include <iostream>
#include <vector>
#include <unordered_map>
#include <ciso646>
#include <algorithm>
#include <cmath>
#include <climits>
#include <random>


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
	Graph(const int nodeNum) :G(nodeNum), mat(nodeNum, vector<int>(nodeNum, 0)) {}
	void add_edge(const int from, const int to, int cost) {
		if (cost == 0) return;
		G[from].push_back(to);
		G[to].push_back(from);
		mat[from][to] = cost;
		mat[to][from] = cost;
	}
	void change_edge(const int from, const int to, int cost) {
		if (mat[from][to] != 0) G[from].push_back(to);
		mat[from][to] = cost;
		mat[to][from] = cost;
	}
	const vector<int> & get_to(const int node) const {
		return G[node];
	}
	int size() {
		return G.size();
	}
	const int get_cost(int from, int to) {
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
	bool get_is_checked(int node) { return isChecked[node]; }
	const vector<int>& get_checked_nodes() const { return checkArray; }
	vector<int> get_not_checks() {
		vector<int> res;
		REP(i, isChecked.size()) if (not isChecked[i]) { res.push_back(i); }
		return res;
	}
};

void map_graph(Check& envCheck, int envNode, unordered_map<int, int>& phi, Check& gCheck, int gNode) {
	phi[envNode] = gNode;
	envCheck.check(envNode);
	gCheck.check(gNode);
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
		int score = 0;
		for (auto &to : G.get_to(from)) {
			score += G.get_cost(from, to);
		}
		if (bestFirst.first < score) {
			bestFirst.first = score;
			bestFirst.second = from;
		}
	}
	int center_node = envG.size() / 2;
	if (envG.size() % 2 == 0) center_node -= sqrt(envG.size()) / 2;
	phi[center_node] = bestFirst.second;
	map_graph(envCheck, center_node, phi, gCheck, bestFirst.second);


	REP(i, G.size() - 1) {
		unordered_map<int, vector<int>> envNodes;
		for (auto &n : envCheck.get_checked_nodes()) for (auto &t : envG.get_to(n)) if (not envCheck.get_is_checked(t)) {
					envNodes[t].push_back(n);
				}
		unordered_map<int, pair<int, double>> envScores; // env側でのnodeの評価値 envScores[envTo] = map(gTo, score);
		for (const pair<int, vector<int> > &nodePair : envNodes) { // ↑の処理でやったenv側のノード first:to, second: froms
			unordered_map<int, double> gScores; // scores[to] = score;
			//G側でスコアの精算
			//envCheckedNodeでメモ化可能
			for (const int &envCheckedNode : nodePair.second) {
				int from = phi[envCheckedNode];// envG -> G
				for (int to : gCheck.get_not_checks()) { // unchecked G nodes
					gScores[to] += G.get_cost(from, to);
                    for(int toto: gCheck.get_not_checks()){
                        gScores[to] += double(G.get_cost(to, toto))/10.0;
					}
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
        vector<pair<int, int> > bestNodes;
        for(const auto &a:envScores){
            if(a.second.second == best.second.second){
                bestNodes.push_back({a.first, a.second.first});
			}
		}
        pair<int, int> selectedNode = bestNodes[random.random(bestNodes.size())]; //first: envNode, second: gNode
		map_graph(envCheck, selectedNode.first, phi, gCheck, selectedNode.second);
	}
	for (auto &a : phi) {
		cout << a.second + 1 << " " << a.first + 1 << endl;
	}
}