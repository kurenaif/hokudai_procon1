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
#include <chrono>


#define FOR(i,a,b) for (int i=(a);i<(b);i++)
#define RFOR(i,a,b) for (int i=(b)-1;i>=(a);i--)
#define REP(i,n) for (int i=0;i<(n);i++)
#define RREP(i,n) for (int i=(n)-1;i>=0;i--)


using namespace std;

constexpr double limit = 9800;

struct pairhash {
public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U> &x) const
  {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

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
		edgeSum[from] += cost;
		edgeSum[to] += cost;
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
	const long get_edge_sum(const int node) const {
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
vector<State> score(const Graph& G, const Graph& envG, const State& state, int size){
	const unordered_map<int, int>& phi = state.phi;
	const Check& envCheck = state.envCheck;
	const Check& gCheck = state.gCheck;
    if(phi.size() == G.size()) return {};
    vector<State> res;
	unordered_map<int, vector<int>> envNodes;
	for (auto &n : envCheck.get_checked_nodes()) for (auto &t : envG.get_to(n)) if (not envCheck.get_is_checked(t)) {
				envNodes[t].push_back(n);
			}
	unordered_map<pair<int, int> , int, pairhash> envScores; // first.first: envG first.second: G, second: score
	for (const pair<int, vector<int> > &nodePair : envNodes) { // ↑の処理でやったenv側のノード first:to, second: froms
		unordered_map<int, int> gScores; // scores[to] = score;
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
		for(const pair<int, int> &p: gScores){
			envScores[{nodePair.first, p.first}] = p.second;
		}
	}
	//env側で最大値
	vector<pair<double, pair<int, int> > > bestNodes;
	for(const auto &a:envScores){
		bestNodes.push_back({a.second, {a.first.first, a.first.second}});
	}
	sort(bestNodes.begin(), bestNodes.end());
	reverse(bestNodes.begin(), bestNodes.end());
	//map graph
	int temp = min(size, int(bestNodes.size()));
    REP(i,temp){
        State s = state;
		map_graph(s.envCheck, bestNodes[i].second.first, s.phi, s.gCheck, bestNodes[i].second.second);
		s.score = bestNodes[i].first;
		res.push_back(s);
	}
	return std::move(res);
}

bool TimeCheck(const chrono::system_clock::time_point start, const double limit){
		const chrono::system_clock::time_point end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		if(elapsed > limit) return false;
		return true;
}

int main(void) {
	chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間
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
		end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		if(elapsed > limit) break;
			bestFirst.second = from;
		}
	}
	int center_node = envG.size() / 2;
	if (envG.size() % 2 == 0) center_node -= sqrt(envG.size()) / 2;
	map_graph(envCheck, center_node, phi, gCheck, bestFirst.second);

	vector<priority_queue<State> > qState(G.size());
	qState[0].push(State(phi, envCheck, gCheck, 0));
	int chokudaiWidth = 1;
	while(TimeCheck(start, 9000)){
		for(int t=0;t < G.size()-1;++t){
			for(int i=0;i<chokudaiWidth;++i){
				if(qState[t].empty()) break;
				State nowState = qState[t].top(); qState[t].pop();
				for(auto &nextState: score(G, envG, nowState, 20)){
					qState[t+1].push(nextState);
				}
			}
			if(not TimeCheck(start, 9000)) break;
		}
	}
	State res = qState.back().top();
	phi = std::move(res.phi);

	priority_queue<pair<int, int> > nodePotential; // first: score, second: envNode
	vector<int> scoreMemo(envG.size()); // scoreMemo[to] = score;
	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	//各ノードが増やせる可能性がある量を列挙
	for(pair<int, int> nodeMap: phi){
		end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		if(elapsed > limit) break;
		int envFrom = nodeMap.first;
		int gFrom = phi[envFrom];
		int score = G.get_edge_sum(gFrom);
		for(const int& envTo: envG.get_to(envFrom)){
			if(phi.count(envTo) != 0){
				int gTo = phi[envTo];
				score -= G.get_cost(gFrom, gTo);
				scoreMemo[envFrom] += G.get_cost(gFrom, gTo);
			}
		}
		nodePotential.push({score, envFrom});
	}

	while(true){
		end = std::chrono::system_clock::now();  // 計測終了時間
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		if(elapsed > limit) break;
		pair<int, int> potential = nodePotential.top(); nodePotential.pop(); //first: score, second: envNode
		int scoreDiff = 0;
		int curNode = potential.second;
		int curPenalty = scoreMemo[curNode]; // cur側を取り除いた時に発生する負債
		pair<int, int> maxScore; //first: score, second: tar
		REP(tarNode,envG.size()){
			end = std::chrono::system_clock::now();  // 計測終了時間
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
			if(elapsed > limit) break;
			int tarPenalty = scoreMemo[tarNode]; // tar側を取り除いた時に発生する負債
			int score = 0;
			for(int to:envG.get_to(tarNode)){ //入れ替えた先がtar側の計算
				if(phi.count(curNode) and phi.count(to)) score += G.get_cost(phi[curNode], phi[to]);
			}
			for(int to:envG.get_to(curNode)){ //入れ替えた先がfrom側の計算
				if(phi.count(tarNode) and phi.count(to)) score += G.get_cost(phi[tarNode], phi[to]);
			}
			score -= curPenalty;
			score -= tarPenalty;
			if(maxScore.first < score){
				maxScore.first = score;
				maxScore.second = tarNode;
			}
		}
		end = std::chrono::system_clock::now();  // 計測終了時間
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		if(elapsed > limit) break;
		//ToDo swapした結果をpriority_queueにpushする
		if(maxScore.first > 0){
			if(not phi.count(curNode)){
				phi[curNode] = phi[maxScore.second];
				phi.erase(maxScore.second);
			}
			else if(not phi.count(maxScore.second)){
				phi[maxScore.second] = phi[curNode];
				phi.erase(curNode);
			}
			else {
				swap(phi[curNode], phi[maxScore.second]);
			}
 
			int envFrom = curNode;
			if(phi.count(envFrom)){
				int gFrom = phi[envFrom];
				int score = G.get_edge_sum(gFrom);
				for(const int& envTo: envG.get_to(envFrom)){
					if(phi.count(envTo) != 0){
						int gTo = phi[envTo];
						score -= G.get_cost(gFrom, gTo);
						scoreMemo[envFrom] += G.get_cost(gFrom, gTo);
					}
				}
				nodePotential.push({score, envFrom});
			}
 
			envFrom = maxScore.second;
			if(phi.count(envFrom)){
				int gFrom = phi[envFrom];
				int score = G.get_edge_sum(gFrom);
				for(const int& envTo: envG.get_to(envFrom)){
					if(phi.count(envTo) != 0){
						int gTo = phi[envTo];
						score -= G.get_cost(gFrom, gTo);
						scoreMemo[envFrom] += G.get_cost(gFrom, gTo);
					}
				}
				nodePotential.push({score, envFrom});
			}
		}
	}
	

	for (auto &a : phi) {
		cout << a.second + 1 << " " << a.first + 1 << endl;
	}
}
