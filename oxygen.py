import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

class OxygenTransportModelWithHbInput:
    def __init__(self):
        # 基本生理参数
        self.Hb_baseline = 140.0  # 基线血红蛋白浓度 g/L
        self.L = 1.251   # 氧-血红蛋白解离曲线参数
        self.k = 0.0676  # 1/mmHg
        self.m = 17.71   # mmHg
        self.b = -0.274  # 偏移量
        self.O2_capacity = 1.34  # mL O2/g Hb, 血红蛋白氧容量
        self.alpha_p = 0.0031  # 氧气在血浆中的溶解系数
        
        # 扩散相关参数
        self.Dm = 1.5e-5  # 扩散系数
        self.Am = 1.35e-6  # 单个红细胞表面积 cm^2
        self.dm = 2.5e-5  # 红细胞膜厚度 cm
        self.RBC_count = 5e6  # 每毫升血液中红细胞数量
        
        # 初始条件
        self.PO2_alveoli = 100.0  # 肺泡氧分压 mmHg
        self.PO2_tissue = 30.0   # 组织氧分压 mmHg
        self.PO2_plasma_initial = 95.0  # 初始血浆氧分压 mmHg
        self.PO2_RBC_initial = 98.0     # 初始红细胞内氧分压 mmHg
        self.Hb_initial = 140.0   # 初始血红蛋白浓度 g/L
        
        # 血红蛋白动力学参数
        self.Hb_input_rate = 0.0  # 血红蛋白输入速率 g/L/hour
        self.Hb_clearance_rate = 0.02  # 血红蛋白清除率 1/hour (半衰期约35小时)
        
    def oxygen_hemoglobin_dissociation(self, PO2):
        """计算给定PO2下的氧饱和度(SaO2)"""
        return self.L / (1 + np.exp(-self.k * (PO2 - self.m))) + self.b
    
    def inverse_oxygen_hemoglobin_dissociation(self, SaO2):
        """从SaO2计算PO2"""
        return self.m + np.log(self.L / (SaO2 - self.b) - 1) / (-self.k)
    
    def calculate_O2Hb(self, SaO2, Hb):
        """计算氧合血红蛋白浓度"""
        return (SaO2 / 100.0) * Hb
    
    def calculate_HHb(self, SaO2, Hb):
        """计算脱氧血红蛋白浓度"""
        return (1 - SaO2 / 100.0) * Hb
    
    def calculate_plasma_O2(self, PO2):
        """计算血浆中溶解的氧气量"""
        return self.alpha_p * PO2
    
    def calculate_total_O2(self, SaO2, PO2, Hb):
        """计算总氧含量 (Hb结合 + 溶解)"""
        O2Hb_content = self.O2_capacity * Hb * (SaO2 / 100.0)
        dissolved_O2 = self.alpha_p * PO2
        return O2Hb_content + dissolved_O2
    
    def diffusion_flux(self, PO2_RBC, PO2_plasma):
        """计算从红细胞到血浆的氧气扩散通量"""
        return self.Dm * self.Am * (PO2_RBC - PO2_plasma) / self.dm
    
    def hb_input_function(self, t, input_profile=None):
        """
        血红蛋白输入函数
        input_profile: 可选的自定义输入配置文件 [(时间, 速率)...]
        """
        if input_profile is not None:
            # 查找当前时间点的输入速率
            for time_point, rate in sorted(input_profile):
                if t < time_point:
                    return rate
            # 如果t大于所有时间点，使用最后一个速率
            return input_profile[-1][1]
        
        # 默认使用恒定输入速率
        return self.Hb_input_rate
    
    def system_equations(self, y, t, params):
        """扩展系统微分方程 - 增加了血红蛋白动力学"""
        PO2_RBC, PO2_plasma, Hb = y
        blood_flow, tissue_consumption, alveolar_exchange, input_profile = params
        
        # 计算氧饱和度
        SaO2 = self.oxygen_hemoglobin_dissociation(PO2_RBC)
        
        # 红细胞与血浆之间的氧气扩散
        J_diff = self.diffusion_flux(PO2_RBC, PO2_plasma) * self.RBC_count
        
        # 血浆中的氧气变化率 = 扩散 + 肺部交换 - 组织消耗
        dPO2_plasma_dt = (J_diff / self.alpha_p  
                         + alveolar_exchange * (self.PO2_alveoli - PO2_plasma)  
                         - tissue_consumption * (PO2_plasma - self.PO2_tissue) / self.alpha_p)
        
        # 红细胞中的氧气变化率 - 考虑血红蛋白浓度
        delta = 0.1
        SaO2_plus = self.oxygen_hemoglobin_dissociation(PO2_RBC + delta)
        buffer_effect = (SaO2_plus - SaO2) / delta
        
        # 氧分压变化率与血红蛋白浓度成反比
        dPO2_RBC_dt = -J_diff / (self.alpha_p * buffer_effect * (Hb / self.Hb_baseline))
        
        # 血红蛋白浓度动力学: 输入 - 清除
        Hb_input = self.hb_input_function(t, input_profile)
        dHb_dt = Hb_input - self.Hb_clearance_rate * Hb
        
        return [dPO2_RBC_dt, dPO2_plasma_dt, dHb_dt]
    
    def simulate(self, simulation_time=24, time_step=0.1, input_profile=None):
        """
        运行模拟，返回结果
        simulation_time: 模拟总时长（小时）
        time_step: 时间步长（小时）
        input_profile: 自定义血红蛋白输入配置文件 [(时间, 速率)...]
        """
        # 设置模拟参数
        blood_flow = 0.05      # 血流速率因子
        tissue_consumption = 0.03  # 组织氧消耗因子
        alveolar_exchange = 0.08   # 肺泡氧交换因子
        
        params = (blood_flow, tissue_consumption, alveolar_exchange, input_profile)
        t = np.arange(0, simulation_time, time_step)
        y0 = [self.PO2_RBC_initial, self.PO2_plasma_initial, self.Hb_initial]  # 初始条件
        
        # 求解微分方程
        solution = odeint(self.system_equations, y0, t, args=(params,))
        
        # 提取结果
        PO2_RBC_values = solution[:, 0]
        PO2_plasma_values = solution[:, 1]
        Hb_values = solution[:, 2]
        SaO2_values = np.array([self.oxygen_hemoglobin_dissociation(po2) for po2 in PO2_RBC_values])
        O2Hb_values = np.array([self.calculate_O2Hb(sao2, hb) 
                               for sao2, hb in zip(SaO2_values, Hb_values)])
        HHb_values = np.array([self.calculate_HHb(sao2, hb) 
                               for sao2, hb in zip(SaO2_values, Hb_values)])
        total_O2_values = np.array([self.calculate_total_O2(sao2, po2, hb) 
                                   for sao2, po2, hb in zip(SaO2_values, PO2_plasma_values, Hb_values)])
        
        return {
            'time': t,
            'PO2_RBC': PO2_RBC_values,
            'PO2_plasma': PO2_plasma_values,
            'Hb': Hb_values,
            'SaO2': SaO2_values,
            'O2Hb': O2Hb_values,
            'HHb': HHb_values,
            'total_O2': total_O2_values
        }
    
    def plot_results(self, results):
        """绘制模拟结果"""
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        
        # 血红蛋白浓度
        axs[0].plot(results['time'], results['Hb'])
        axs[0].set_ylabel('血红蛋白浓度 (g/L)')
        axs[0].set_title('血红蛋白动态变化')
        axs[0].grid(True)
        
        # 氧分压
        axs[1].plot(results['time'], results['PO2_RBC'], label='PO2 in RBC')
        axs[1].plot(results['time'], results['PO2_plasma'], label='PO2 in Plasma')
        axs[1].set_ylabel('PO2 (mmHg)')
        axs[1].set_title('氧分压变化')
        axs[1].legend()
        axs[1].grid(True)
        
        # 血红蛋白状态
        axs[2].plot(results['time'], results['O2Hb'], label='O2Hb')
        axs[2].plot(results['time'], results['HHb'], label='HHb')
        axs[2].plot(results['time'], results['SaO2'], label='SaO2 (%)')
        axs[2].set_ylabel('浓度 (g/L) / 饱和度 (%)')
        axs[2].set_title('血红蛋白氧合状态')
        axs[2].legend()
        axs[2].grid(True)
        
        # 总氧含量
        axs[3].plot(results['time'], results['total_O2'])
        axs[3].set_xlabel('时间 (小时)')
        axs[3].set_ylabel('总氧含量 (ml O2/L)')
        axs[3].set_title('血液中总氧含量')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.show()

    def simulate_transfusion(self, simulation_time=48, transfusion_time=6, 
                            transfusion_duration=2, transfusion_amount=30):
        """
        模拟输血场景
        simulation_time: 总模拟时间 (小时)
        transfusion_time: 输血开始时间 (小时)
        transfusion_duration: 输血持续时间 (小时)
        transfusion_amount: 输血增加的血红蛋白总量 (g/L)
        """
        # 计算输血期间的输入速率
        transfusion_rate = transfusion_amount / transfusion_duration
        
        # 创建输入配置文件
        input_profile = [
            (0, 0),  # 初始无输入
            (transfusion_time, transfusion_rate),  # 开始输血
            (transfusion_time + transfusion_duration, 0)  # 结束输血
        ]
        
        # 运行模拟
        results = self.simulate(simulation_time=simulation_time, 
                               time_step=0.1, 
                               input_profile=input_profile)
        
        # 绘制结果
        self.plot_results(results)
        return results
    
    def simulate_continuous_production(self, simulation_time=72, 
                                      baseline_production=0.5, 
                                      stimulated_production=1.5,
                                      stimulation_time=24):
        """
        模拟红细胞生成增加（如促红细胞生成素作用）
        simulation_time: 总模拟时间 (小时)
        baseline_production: 基础血红蛋白生成率 (g/L/小时)
        stimulated_production: 刺激后血红蛋白生成率 (g/L/小时)
        stimulation_time: 刺激开始时间 (小时)
        """
        # 创建输入配置文件
        input_profile = [
            (0, baseline_production),  # 基础生成率
            (stimulation_time, stimulated_production)  # 刺激后生成率
        ]
        
        # 运行模拟
        results = self.simulate(simulation_time=simulation_time, 
                               time_step=0.1, 
                               input_profile=input_profile)
        
        # 绘制结果
        self.plot_results(results)
        return results
    
    def compare_scenarios(self):
        """比较不同血红蛋白输入场景下的氧气传输"""
        # 基线场景 - 无额外输入
        self.Hb_input_rate = 0
        baseline_results = self.simulate(simulation_time=48)
        
        # 单次输血场景
        transfusion_results = self.simulate_transfusion(simulation_time=48)
        
        # 持续生成增加场景
        production_results = self.simulate_continuous_production(simulation_time=48)
        
        # 对比关键指标
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # 血红蛋白浓度比较
        axs[0].plot(baseline_results['time'], baseline_results['Hb'], 
                   label='基线', linestyle='-')
        axs[0].plot(transfusion_results['time'], transfusion_results['Hb'], 
                   label='单次输血', linestyle='--')
        axs[0].plot(production_results['time'], production_results['Hb'], 
                   label='生成增加', linestyle=':')
        axs[0].set_ylabel('血红蛋白浓度 (g/L)')
        axs[0].set_title('不同场景下的血红蛋白变化')
        axs[0].legend()
        axs[0].grid(True)
        
        # 氧合血红蛋白比较
        axs[1].plot(baseline_results['time'], baseline_results['O2Hb'], 
                   label='基线', linestyle='-')
        axs[1].plot(transfusion_results['time'], transfusion_results['O2Hb'], 
                   label='单次输血', linestyle='--')
        axs[1].plot(production_results['time'], production_results['O2Hb'], 
                   label='生成增加', linestyle=':')
        axs[1].set_ylabel('氧合血红蛋白浓度 (g/L)')
        axs[1].set_title('不同场景下的氧合血红蛋白变化')
        axs[1].legend()
        axs[1].grid(True)
        
        # 总氧含量比较
        axs[2].plot(baseline_results['time'], baseline_results['total_O2'], 
                   label='基线', linestyle='-')
        axs[2].plot(transfusion_results['time'], transfusion_results['total_O2'], 
                   label='单次输血', linestyle='--')
        axs[2].plot(production_results['time'], production_results['total_O2'], 
                   label='生成增加', linestyle=':')
        axs[2].set_xlabel('时间 (小时)')
        axs[2].set_ylabel('总氧含量 (ml O2/L)')
        axs[2].set_title('不同场景下的总氧含量变化')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
def run_simulation():
    # 创建模型
    model = OxygenTransportModelWithHbInput()
    
    print("模拟单次输血场景...")
    model.simulate_transfusion(transfusion_time=6, transfusion_amount=40)
    
    print("模拟红细胞生成增加场景...")
    model.simulate_continuous_production(stimulation_time=24)
    
    print("比较不同场景...")
    model.compare_scenarios()
    
    print("模拟完成!")

if __name__ == "__main__":
    run_simulation()