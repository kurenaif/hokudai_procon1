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
#include <cassert>


#define FOR(i,a,b) for (int i=(a);i<(b);i++)
#define RFOR(i,a,b) for (int i=(b)-1;i>=(a);i--)
#define REP(i,n) for (int i=0;i<(n);i++)
#define RREP(i,n) for (int i=(n)-1;i>=0;i--)


using namespace std;

constexpr double limit = 9500;


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
	Check(int node) :checkArray(), isChecked(node, false) {
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
    int before = phi.size();
	phi[envNode] = gNode;
	envCheck.check(envNode);
	gCheck.check(gNode);
	int after = phi.size();
    assert(before != after);
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
map<State, vector<State>> stateMemo;// state, size: value vector<State>
vector<State> score(const Graph& G, const Graph& envG, const State& state, int size){
	if(stateMemo.count(state) == 1) {
		return stateMemo[state];
	}
	const unordered_map<int, int>& phi = state.phi;
	const Check& envCheck = state.envCheck;
	const Check& gCheck = state.gCheck;
	if(phi.size() == G.size()) return {};
	unordered_map<int, vector<int>> envNodes; // to側からfromを登録する
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
	int temp = min(size, int(bestNodes.size()));
	vector<State> res(temp, state);
	REP(i,temp){
		State& s = res[i];
		map_graph(s.envCheck, bestNodes[i].second.first, s.phi, s.gCheck, bestNodes[i].second.second);
		s.score = bestNodes[i].first;
	}
	stateMemo[state] = res;
	return std::move(res);
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

	queue<State> que;
	que.push(State(phi, envCheck, gCheck, 0));

	vector<State> lasts;
	int width = 2;
	int nextWidth = 2;
	int cnt = 0;
	while(not que.empty()) {
        ++cnt;
		//cerr << "cnt:" << cnt << endl;
		State state = que.front(); que.pop();
		if(state.phi.size() == G.size()){
			lasts.push_back(state);
			continue;
		}
		vector<State> next = score(G, envG, state, width);
		long scoreElapsed = 0;
		vector<int> points(next.size());
		REP(i,next.size()){
			auto scoreStart = std::chrono::system_clock::now();  // 計測終了時間
			points[i] = DFS(G, envG, next[i], nextWidth, 1);
			auto scoreEnd = std::chrono::system_clock::now();  // 計測終了時
            scoreElapsed = max(scoreElapsed, static_cast<long>(chrono::duration_cast<std::chrono::microseconds>(scoreEnd-scoreStart).count())); //処理に要した時間をミリ秒に変換
		}
		int index = max_element(points.begin(), points.end()) - points.begin();
		//cerr << "phi cnt:" << state.phi.size() << " " << next[index].phi.size() << endl;
		que.push(next[index]);
        if(scoreElapsed < 100){
			width = G.size();
			nextWidth = G.size();
		}
		auto end = std::chrono::system_clock::now();  // 計測終了時間
		double restTime = 9500000 - std::chrono::duration_cast<std::chrono::microseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
		double restCount = restTime/(scoreElapsed);
		//cerr << "scoreElapsed:" << scoreElapsed << endl;
		//cerr << "restTime:" << restTime << endl;
		//cerr << "restCount:" << restCount << endl;
        width = min(int(sqrt(restCount/(G.size() - cnt))), G.size()-1);
        nextWidth = width;
		//cerr << "width:" << width << endl;
		//cerr << "nextWidth:" << nextWidth << endl;
	}
	State res = *max_element(lasts.begin(), lasts.end());
	phi = std::move(res.phi);

	priority_queue<pair<int, int> > nodePotential; // first: score, second: envNode
	vector<int> scoreMemo(envG.size()); // scoreMemo[to] = score;
	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	//各ノードが増やせる可能性がある量を列挙
	for(pair<int, int> nodeMap: phi){
		auto end = std::chrono::system_clock::now();  // 計測終了時間
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
