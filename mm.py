import pandas as pd
from pytrends.request import TrendReq
import time
import os

df = pd.read_csv('data.csv')

ANCHOR_KEYWORD = "Dancing with the Stars"

# 赛季时间表配置
season_configs = {
    # 1:  '2005-06-01 2005-07-06',
    # 2:  '2006-01-05 2006-02-26',
    # 3:  '2006-09-12 2006-11-15',
    # 4:  '2007-03-19 2007-05-22',
    # 5:  '2007-09-24 2007-11-27',
    # 6:  '2008-03-17 2008-05-20',
    # 7:  '2008-09-22 2008-11-25',
    # 8:  '2009-03-09 2009-05-19',
    # 9:  '2009-09-21 2009-11-24',
    # 10: '2010-03-22 2010-05-25',
    # 11: '2010-09-20 2010-11-23',
    # 12: '2011-03-21 2011-05-24',
    # 13: '2011-09-19 2011-11-22',
    # 14: '2012-03-19 2012-05-22',
    # 15: '2012-09-24 2012-11-27',
    # 16: '2013-03-18 2013-05-21',
    # 17: '2013-09-16 2013-11-26',
    # 18: '2014-03-17 2014-05-20',
    # 19: '2014-09-15 2014-11-25',
    # 20: '2015-03-16 2015-05-19',
    # 21: '2015-09-14 2015-11-24',
    # 22: '2016-03-21 2016-05-24',
     23: '2016-09-12 2016-11-22',
    # 24: '2017-03-20 2017-05-23',
    # 25: '2017-09-18 2017-11-21',
    # 26: '2018-04-30 2018-05-21',
    #27: '2018-09-24 2018-11-19',
    #28: '2019-09-16 2019-11-25',
    #29: '2020-09-14 2020-11-23',
    #30: '2021-09-20 2021-11-22',
    31: '2022-09-19 2022-11-21',
    #32: '2023-09-26 2023-12-05',
    #33: '2024-09-17 2024-11-26',
    #34: '2025-09-16 2025-11-25' 
}


def fetch_season_trends(season_num, date_range, contestants):
    """
    抓取特定赛季、特定时间段的选手数据并进行归一化
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    
    final_df = pd.DataFrame()
    
    base_batch = contestants[:4] + [ANCHOR_KEYWORD]
    print(f"  [Season {season_num}] 正在抓取基准批次: {base_batch}")
    
    try:
        pytrends.build_payload(base_batch, cat=0, timeframe=date_range, geo='US', gprop='')
        base_data = pytrends.interest_over_time()
        
        if base_data.empty:
            print(f"  [Warning] Season {season_num} 基准数据为空，跳过。")
            return None
            
        if 'isPartial' in base_data.columns:
            del base_data['isPartial']
            
        # 保存基准数据
        final_df = base_data.copy()
        
        # 计算基准锚点的平均值 (作为标准 1.0)
        base_anchor_mean = base_data[ANCHOR_KEYWORD].mean()
        
        # 循环剩余选手 (从第5个开始，每4个一组)
        for i in range(4, len(contestants), 4):
            batch = contestants[i : i+4]
            current_keywords = batch + [ANCHOR_KEYWORD]
            
            print(f"  [Season {season_num}] 正在抓取批次: {current_keywords}")
            time.sleep(30) # 防封锁延时
            
            pytrends.build_payload(current_keywords, cat=0, timeframe=date_range, geo='US', gprop='')
            current_data = pytrends.interest_over_time()
            
            if 'isPartial' in current_data.columns:
                del current_data['isPartial']
            
            # 归一化计算
            curr_anchor_mean = current_data[ANCHOR_KEYWORD].mean()
            if curr_anchor_mean > 0:
                factor = base_anchor_mean / curr_anchor_mean
            else:
                factor = 0
                
            # 将当前批次选手的数据校准后加入 final_df
            for person in batch:
                if person in current_data.columns:
                    final_df[person] = current_data[person] * factor
        
        return final_df

    except Exception as e:
        print(f"  [Error] Season {season_num} 抓取失败: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists('dwts_results'):
        os.makedirs('dwts_results')

    for season_num, date_range in season_configs.items():
        if not date_range:
            continue
            
        print(f"\n=== 开始处理第 {season_num} 季 (时间: {date_range}) ===")
        # 筛选 season 列等于当前循环号，并提取 celebrity_name
        season_contestants = df[df['season'] == season_num]['celebrity_name'].unique().tolist()
        
        if not season_contestants:
            print(f"  [Info] CSV中没有找到第 {season_num} 季的选手数据。")
            continue
            
        print(f"  选手名单 ({len(season_contestants)}人): {season_contestants}")
        
        result_df = fetch_season_trends(season_num, date_range, season_contestants)
        
        if result_df is not None:
            filename = f"dwts_results/season_{season_num}_trends.csv"
            result_df.to_csv(filename)
            print(f"  [Success] 结果已保存至: {filename}")
        
    print("\n所有已配置的赛季处理完毕！")