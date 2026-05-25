from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import osmnx as ox
import networkx as nx
import math
import random

app = FastAPI()

# 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 全局變數：用於地圖快取機制 ===
G = None
last_download_coords = None
current_net_type = None 
last_fetch_radius = 0  

# 接收前端資料的格式
class RunRequest(BaseModel):
    weight_kg: float
    target_dist_km: float
    start_lat: float
    start_lng: float

@app.post("/calculate_route")
def calculate_route(req: RunRequest):
    global G, last_download_coords, current_net_type, last_fetch_radius
    
    # 防呆機制：限制最大距離為全馬範圍
    if req.target_dist_km > 45:
        return {"status": "error", "message": "距離請勿超過 45 公里限制。"}
        
    target_dist_m = req.target_dist_km * 1000
    
    # 🌟 智慧路網切換 (超過 15 公里自動切換為汽車路網，確保連通性)
    chosen_net_type = 'walk' if req.target_dist_km <= 15 else 'drive'
    
    # 🌟 動態半徑優化 (依據距離自動調整下載範圍)
    if req.target_dist_km <= 10:
        fetch_radius_m = (target_dist_m / 2.0) * 1.3
    elif req.target_dist_km <= 25:
        fetch_radius_m = (target_dist_m / 2.0) * 1.15
    else:
        fetch_radius_m = (target_dist_m / 2.0) * 1.05 
        
    fetch_radius_m = max(fetch_radius_m, 2500.0)

    # 判斷是否需要重新下載圖資
    need_reload = True
    if last_download_coords is not None and G is not None and current_net_type == chosen_net_type:
        lat_diff = req.start_lat - last_download_coords[0]
        lng_diff = req.start_lng - last_download_coords[1]
        approx_distance_m = math.sqrt(lat_diff**2 + lng_diff**2) * 111000
        
        # 確保快取地圖的半徑足夠大
        if approx_distance_m < 2000 and fetch_radius_m <= last_fetch_radius:
            need_reload = False

    if need_reload:
        print(f"🌐 [動態路網] 正在下載專屬街道圖資...")
        print(f"📍 類型: {chosen_net_type} | 半徑: {round(fetch_radius_m)}公尺")
        try:
            G = ox.graph_from_point((req.start_lat, req.start_lng), dist=fetch_radius_m, network_type=chosen_net_type)
            last_download_coords = (req.start_lat, req.start_lng)
            current_net_type = chosen_net_type
            last_fetch_radius = fetch_radius_m
            print("✅ 圖資載入完成！")
        except Exception as e:
            return {"status": "error", "message": f"圖資下載逾時，請縮短距離或稍後再試: {str(e)}"}

    try:
        # 尋找最近路口
        start_node = ox.distance.nearest_nodes(G, X=req.start_lng, Y=req.start_lat)
        half_dist_m = target_dist_m / 2.0
        
        # 尋找折返點候選人
        lengths = nx.single_source_dijkstra_path_length(G, start_node, cutoff=half_dist_m * 1.15, weight='length')
        candidates = [n for n, dist in lengths.items() if half_dist_m * 0.85 <= dist <= half_dist_m * 1.15]
        
        if not candidates:
            sorted_nodes = sorted(lengths.items(), key=lambda x: x[1])
            candidates = [n for n, dist in sorted_nodes[-5:]] if sorted_nodes else [start_node]
            
        # 🌟 A* 演算法的啟發函數 (直線距離)
        def astar_heuristic(u, v):
            return math.sqrt((G.nodes[u]['y'] - G.nodes[v]['y'])**2 + (G.nodes[u]['x'] - G.nodes[v]['x'])**2) * 111000

        best_route_nodes = []
        best_diff = float('inf')
        actual_total_dist_m = 0
        
        # 多重路線海選測試
        sample_count = 1 if req.target_dist_km > 20 else min(3, len(candidates))
        test_candidates = random.sample(candidates, sample_count)
        
        for mid_node in test_candidates:
            path_out = []
            try:
                # 嘗試去程
                path_out = nx.astar_path(G, source=start_node, target=mid_node, heuristic=astar_heuristic, weight='length')
                dist_out = sum(G[u][v][0]['length'] for u, v in zip(path_out[:-1], path_out[1:]))
                
                # 施加去程懲罰
                for u, v in zip(path_out[:-1], path_out[1:]):
                    if G.has_edge(u, v):
                        for key in G[u][v]:
                            G[u][v][key]['original_length'] = G[u][v][key]['length']
                            G[u][v][key]['length'] *= 100
                    if G.has_edge(v, u):
                        for key in G[v][u]:
                            G[v][u][key]['original_length'] = G[v][u][key]['length']
                            G[v][u][key]['length'] *= 100

                # 嘗試回程
                try:
                    path_back = nx.astar_path(G, source=mid_node, target=start_node, heuristic=astar_heuristic, weight='length')
                    dist_back = sum(G[u][v][0].get('original_length', G[u][v][0]['length']) for u, v in zip(path_back[:-1], path_back[1:]))
                    
                    total_dist = dist_out + dist_back
                    diff = abs(total_dist - target_dist_m)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_route_nodes = path_out[:-1] + path_back
                        actual_total_dist_m = total_dist
                except nx.NetworkXNoPath:
                    pass 
                    
            except nx.NetworkXNoPath:
                pass 
                
            finally:
                # 🌟 致命 Bug 修復：強制還原地圖權重！
                if path_out:
                    for u, v in zip(path_out[:-1], path_out[1:]):
                        if G.has_edge(u, v):
                            for key in G[u][v]:
                                if 'original_length' in G[u][v][key]:
                                    G[u][v][key]['length'] = G[u][v][key]['original_length']
                        if G.has_edge(v, u):
                            for key in G[v][u]:
                                if 'original_length' in G[v][u][key]:
                                    G[v][u][key]['length'] = G[v][u][key]['original_length']

        if not best_route_nodes:
            return {"status": "error", "message": "此區域路網破碎或單行道過多，無法生成環狀路線。請稍微調整距離或更換起點。"}

        # 組合路線並擷取真實道路形狀 (Geometry)
        route_coords = []
        for i in range(len(best_route_nodes) - 1):
            u = best_route_nodes[i]
            v = best_route_nodes[i+1]
            edge_data = G[u][v][0] 
            if 'geometry' in edge_data:
                for lon, lat in list(edge_data['geometry'].coords):
                    route_coords.append({"lat": lat, "lng": lon})
            else:
                route_coords.append({"lat": G.nodes[u]['y'], "lng": G.nodes[u]['x']})
                
        last_node = best_route_nodes[-1]
        route_coords.append({"lat": G.nodes[last_node]['y'], "lng": G.nodes[last_node]['x']})

        # 計算最後實際的精確公里數與卡路里
        actual_dist_km = round(actual_total_dist_m / 1000.0, 2)
        actual_calories = round(req.weight_kg * actual_dist_km * 1.036)

        return {
            "status": "success",
            "estimated_calories": actual_calories,
            "actual_dist_km": actual_dist_km,
            "route": route_coords,
            "message": f"成功為您規劃 {actual_dist_km} 公里環狀路線！"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"系統核心運算失敗: {str(e)}"}

# 啟動引擎
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)