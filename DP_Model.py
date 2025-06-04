from functools import lru_cache
from timeit import default_timer as timer


# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------

def daily_to_monthly(advertiser_number, budget):
    """
    This function is no use for now, 
    only need this function if the input is daily-wise rather than month-wise
    Assume the given budget is already filtered by threshold
    Convert data (length 365) leap_year data (length 366) into monthly data (length 12).
    
    :param advertiser_number: list of length 365/366, advertisers arriving each day
    :param budget: list of length 365/366, average budget (per advertiser) for each day

    :return: (monthly_advertisers, monthly_budget) each of length 12
    """
    # Days in each month (non-leap year)
    leap_year = len(advertiser_number) == 366
    if leap_year == False:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    monthly_advertisers = [0] * 12
    monthly_budget = [0] * 12
    
    day_offset = 0
    
    for m in range(12):
        total_adv_this_month = 0     
        weighted_budget_sum = 0.0  

        for d in range(days_in_month[m]):
            day_index = day_offset + d
            adv_d = advertiser_number[day_index]
            bud_d = budget[day_index]
            
            total_adv_this_month += adv_d
            weighted_budget_sum += adv_d * bud_d
        
        # Store the total number of advertisers for month m
        monthly_advertisers[m] = total_adv_this_month
        
        # Compute the weighted-average budget for month m
        if total_adv_this_month > 0:
            monthly_budget[m] = weighted_budget_sum / total_adv_this_month
        else:
            monthly_budget[m] = 0.0
        
        day_offset += days_in_month[m]
    
    return monthly_advertisers, monthly_budget

def assign_from_waiting(wait_tuple, number_to_assign, WAIT_MONTHS):
    """
    Assign up to 'number_to_assign' advertisers from the waiting pool,
    starting from the oldest waiting slot wait_tuple[-1] backward.

    Returns (updated_wait_tuple, assigned_count).
    """
    w = list(wait_tuple)
    assigned = 0
    i = WAIT_MONTHS - 1
    while assigned < number_to_assign and i >= 0:
        can_take = min(w[i], number_to_assign - assigned)
        w[i] -= can_take
        assigned += can_take
        i -= 1
    return tuple(w), assigned

def advance_waiting(wait_tuple, WAIT_MONTHS):
    """
    Shift waiting forward by 1 month:
      - advertisers in wait_tuple[0] move to wait_tuple[1],
      - the oldest wait_tuple[-1] are effectively lost if not assigned.
    """
    w = list(wait_tuple)
    w[WAIT_MONTHS - 1] = w[0]
    w[0] = 0
    return tuple(w)

def advance_service(service_tuple, SERVICE_MONTHS):
    """
    Shift service forward by 1 month:
      - advertisers in service_tuple[0] move to service_tuple[1],
      - advertisers in service_tuple[-1] 'graduate' and finish.
      
    Returns (shifted_service, count_of_graduated_advertisers).
    """
    s = list(service_tuple)
    graduated = s[SERVICE_MONTHS - 1]
    s[SERVICE_MONTHS - 1] = s[0]
    s[0] = 0
    return tuple(s), graduated

def advance_pipeline(pipe_tuple):
    """
    Advance agent training pipeline by 1 month:
      - pipeline[0] are the agents that become active this month.

    Returns (shifted_pipeline, newly_active_agents).
    """
    p = list(pipe_tuple)
    newly_active = p[0]
    p[0] = 0
    return tuple(p), newly_active

def next_waiting_state(wait_tuple, arrivals_next_month):
    """
    Put next month's new arrivals into wait_tuple[0].
    """
    w = list(wait_tuple)
    w[0] += arrivals_next_month
    return tuple(w)

# ------------------------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------------------------
@lru_cache(None)
def dp_func(
    m, 
    active_agents, 
    pipeline, 
    waiting, 
    service,
    adv_num_tuple, 
    adv_budget_tuple, 
    average_leftover_budget,
    MONTHS,
    WAIT_MONTHS,
    SERVICE_MONTHS,
    TRAIN_MONTHS,
    CAPACITY_PER_AGENT,
    monthly_agent_cost,
    firing_cost,
    BONUS_RATE,
    max_hire_range,
    max_fire_range
):
    """
    The DP function. Returns the maximum profit achievable from month m to Dec
    given the current state (active agents, pipeline, waiting advertisers, service advertisers).

    :param m: current month (1..12) index
    :param active_agents: number of agents currently active this month
    :param pipeline: tuple of length TRAIN_MONTHS -> how many agents are in each training slot
    :param waiting: tuple of length WAIT_MONTHS  -> how many advertisers have waited 0,1,... months
    :param service: tuple of length SERVICE_MONTHS -> how many advertisers are in their service months
    :param adv_num_tuple: tuple/list of length MONTHS+1 with the #advertisers arriving each month (1-based)
    :param adv_budget_tuple: tuple/list of length MONTHS+1 with the average budget in each month (1-based)
    :param average_leftover_budget: used for leftover bonus after month 12
    :param MONTHS, WAIT_MONTHS, SERVICE_MONTHS, TRAIN_MONTHS: pipeline/wait logic
    :param CAPACITY_PER_AGENT: how many advertisers per agent
    :param monthly_agent_cost: monthly cost for each active agent
    :param firing_cost: cost of firing one agent
    :param BONUS_RATE: 13.5% = 0.135
    :param max_hire_range, max_fire_range: maximum allowed hires/fires in any month

    :return: best (maximum) profit from month m..MONTHS.
    """
    if m > MONTHS:
        # Base case: after month 12, pay out leftover bonuses
        leftover_in_service = sum(service)
        leftover_bonus = leftover_in_service * average_leftover_budget * BONUS_RATE
        return leftover_bonus

    cost_this_month = active_agents * monthly_agent_cost
    already_in_service = sum(service)
    capacity_left = active_agents * CAPACITY_PER_AGENT - already_in_service
    capacity_left = max(0, capacity_left)
    total_waiting = sum(waiting)
    can_assign = min(total_waiting, capacity_left)
    adv_num = adv_num_tuple
    adv_budget = adv_budget_tuple
    best_value = float('-inf')

    # Try all combinations of hire/fire within the allowed range
    for h in range(0, max_hire_range + 1, 10):
        for f in range(0, max_fire_range + 1, 10):
            # We can not fire more than we have
            if f > active_agents:
                continue
            # No need for hire and fire in same month:
            if h > 0 and f > 0:
                continue

            cost_firing = f * firing_cost

            # Place assigned advertisers into service[0]
            updated_waiting, assigned_count = assign_from_waiting(waiting, can_assign, WAIT_MONTHS)
            new_service = list(service)
            new_service[0] += assigned_count
            new_service = tuple(new_service)

            # Check graduate
            shifted_service, graduated = advance_service(new_service, SERVICE_MONTHS)
            bonus_revenue = graduated * adv_budget[m] * BONUS_RATE

            # Profit
            immediate_profit = bonus_revenue - cost_this_month - cost_firing

            # Check agent
            shifted_waiting = advance_waiting(updated_waiting, WAIT_MONTHS)
            pipeline_after_shift, newly_active = advance_pipeline(pipeline)
            next_active = active_agents - f + newly_active
            if next_active < 0:
                continue

            # Next month
            new_pipeline = list(pipeline_after_shift)
            new_pipeline[0] += h
            new_pipeline = tuple(new_pipeline)
            arrival_next_month = adv_num[m + 1] if (m + 1) <= MONTHS else 0
            next_wait = next_waiting_state(shifted_waiting, arrival_next_month)

            # Pass updated states to next month
            future_profit = dp_func(
                m+1,
                next_active,
                new_pipeline,
                next_wait,
                shifted_service,
                adv_num,
                adv_budget,
                average_leftover_budget,
                MONTHS,
                WAIT_MONTHS,
                SERVICE_MONTHS,
                TRAIN_MONTHS,
                CAPACITY_PER_AGENT,
                monthly_agent_cost,
                firing_cost,
                BONUS_RATE,
                max_hire_range,
                max_fire_range
            )

            # Get max profit
            total_profit = immediate_profit + future_profit
            if total_profit > best_value:
                best_value = total_profit

    return best_value

# ------------------------------------------------------------------------------------
# Reconstruct function
# ------------------------------------------------------------------------------------
def reconstruct_path(
    m,
    active_agents,
    pipeline,
    waiting,
    service,
    adv_num_tuple,
    adv_budget_tuple,
    average_leftover_budget,
    MONTHS,
    WAIT_MONTHS,
    SERVICE_MONTHS,
    TRAIN_MONTHS,
    CAPACITY_PER_AGENT,
    monthly_agent_cost,
    firing_cost,
    BONUS_RATE,
    max_hire_range,
    max_fire_range,
    best_agent_plan,
    dp_func_ref
):
    """
    Recursively reconstruct the path (number of agents each month) 
    from the DP decisions. We compare each possible hire/fire decision
    to see which matches the stored optimal value.
    """
    if m > MONTHS:
        return  

    dp_value = dp_func_ref(
        m,
        active_agents,
        pipeline,
        waiting,
        service,
        adv_num_tuple,
        adv_budget_tuple,
        average_leftover_budget,
        MONTHS,
        WAIT_MONTHS,
        SERVICE_MONTHS,
        TRAIN_MONTHS,
        CAPACITY_PER_AGENT,
        monthly_agent_cost,
        firing_cost,
        BONUS_RATE,
        max_hire_range,
        max_fire_range
    )

    best_agent_plan[m] = active_agents

    cost_this_month = active_agents * monthly_agent_cost
    already_in_service = sum(service)
    capacity_left = active_agents * CAPACITY_PER_AGENT - already_in_service
    capacity_left = max(0, capacity_left)
    total_waiting = sum(waiting)
    can_assign = min(total_waiting, capacity_left)

    adv_num = adv_num_tuple
    adv_budget = adv_budget_tuple

    EPS = 1e-9

    for h in range(0, max_hire_range + 1, 10):
        for f in range(0, max_fire_range + 1, 10):
            if f > active_agents:
                continue
            if h > 0 and f > 0:
                continue

            cost_firing = f * firing_cost

            updated_waiting, assigned_count = assign_from_waiting(waiting, can_assign, WAIT_MONTHS)
            new_service = list(service)
            new_service[0] += assigned_count
            new_service = tuple(new_service)

            shifted_service, graduated = advance_service(new_service, SERVICE_MONTHS)
            bonus_revenue = graduated * adv_budget[m] * BONUS_RATE
            immediate_profit = - cost_this_month - cost_firing + bonus_revenue

            shifted_waiting = advance_waiting(updated_waiting, WAIT_MONTHS)
            pipeline_after_shift, newly_active = advance_pipeline(pipeline)

            next_active = active_agents - f + newly_active
            if next_active < 0:
                continue

            new_pipeline = list(pipeline_after_shift)
            new_pipeline[0] += h
            new_pipeline = tuple(new_pipeline)

            arrival_next_month = adv_num[m+1] if (m+1) <= MONTHS else 0
            next_wait = next_waiting_state(shifted_waiting, arrival_next_month)

            future_profit = dp_func_ref(
                m+1,
                next_active,
                new_pipeline,
                next_wait,
                shifted_service,
                adv_num,
                adv_budget,
                average_leftover_budget,
                MONTHS,
                WAIT_MONTHS,
                SERVICE_MONTHS,
                TRAIN_MONTHS,
                CAPACITY_PER_AGENT,
                monthly_agent_cost,
                firing_cost,
                BONUS_RATE,
                max_hire_range,
                max_fire_range
            )

            total_profit = immediate_profit + future_profit
            if abs(total_profit - dp_value) < EPS:
                reconstruct_path(
                    m+1,
                    next_active,
                    new_pipeline,
                    next_wait,
                    shifted_service,
                    adv_num_tuple,
                    adv_budget_tuple,
                    average_leftover_budget,
                    MONTHS,
                    WAIT_MONTHS,
                    SERVICE_MONTHS,
                    TRAIN_MONTHS,
                    CAPACITY_PER_AGENT,
                    monthly_agent_cost,
                    firing_cost,
                    BONUS_RATE,
                    max_hire_range,
                    max_fire_range,
                    best_agent_plan,
                    dp_func_ref
                )
                return

# ------------------------------------------------------------------------------------
# Main solver for the result
# ------------------------------------------------------------------------------------
def solve_monthly_dp_with_leftover_bonus(
    advertiser_number,
    budget,
    agent_salary,
    agent_init,
    max_hire_range,
    max_fire_range
):
    """
    Main solver for DP, also reconstructs the optimal path (agents per month).
    
    :param advertiser_number: list of length 12, advertisers arriving each month (1..12)
    :param budget: list of length 12, average budget for advertisers arriving each month
    :param agent_salary: annual salary for one agent
    :param agent_init: initial number of agents active in month 1
    :param max_hire_range: maximum number of agents that can be hired in any month
    :param max_fire_range: maximum number of agents that can be fired in any month

    :return: (max_profit, best_agent_plan)
    """
    MONTHS = 12
    WAIT_MONTHS = 2
    SERVICE_MONTHS = 2
    TRAIN_MONTHS = 1
    CAPACITY_PER_AGENT = 10
    BONUS_RATE = 0.135

    monthly_agent_cost = agent_salary / 12.0
     # 40% of annual salary for firing
    firing_cost = 0.40 * agent_salary 

    # For leftover bonus after month 12, since this would only happen for Dec.
    average_leftover_budget = budget[11]

    # Init
    adv_num = [0] + advertiser_number[:]  
    adv_budget = [0] + budget[:]

    init_pipeline = tuple([0] * TRAIN_MONTHS)  
    init_waiting  = tuple([0] * WAIT_MONTHS)   
    init_service  = tuple([0] * SERVICE_MONTHS) 

    w_list = list(init_waiting)
    w_list[0] = adv_num[1]
    init_waiting = tuple(w_list)

    # Clear cache
    dp_func.cache_clear()

    # ------------------------------------------------------------------------------------
    # Solve DP
    # ------------------------------------------------------------------------------------
    max_profit = dp_func(
        m=1,
        active_agents=agent_init,
        pipeline=init_pipeline,
        waiting=init_waiting,
        service=init_service,
        adv_num_tuple=tuple(adv_num),
        adv_budget_tuple=tuple(adv_budget),
        average_leftover_budget=average_leftover_budget,
        MONTHS=MONTHS,
        WAIT_MONTHS=WAIT_MONTHS,
        SERVICE_MONTHS=SERVICE_MONTHS,
        TRAIN_MONTHS=TRAIN_MONTHS,
        CAPACITY_PER_AGENT=CAPACITY_PER_AGENT,
        monthly_agent_cost=monthly_agent_cost,
        firing_cost=firing_cost,
        BONUS_RATE=BONUS_RATE,
        max_hire_range=max_hire_range,
        max_fire_range=max_fire_range
    )

    print(f'Max profit: {max_profit}')

    # ------------------------------------------------------------------------------------
    # Reconstruct best plan
    # ------------------------------------------------------------------------------------
    best_agent_plan = [0] * (MONTHS + 1)  
    reconstruct_path(
        m=1,
        active_agents=agent_init,
        pipeline=init_pipeline,
        waiting=init_waiting,
        service=init_service,
        adv_num_tuple=tuple(adv_num),
        adv_budget_tuple=tuple(adv_budget),
        average_leftover_budget=average_leftover_budget,
        MONTHS=MONTHS,
        WAIT_MONTHS=WAIT_MONTHS,
        SERVICE_MONTHS=SERVICE_MONTHS,
        TRAIN_MONTHS=TRAIN_MONTHS,
        CAPACITY_PER_AGENT=CAPACITY_PER_AGENT,
        monthly_agent_cost=monthly_agent_cost,
        firing_cost=firing_cost,
        BONUS_RATE=BONUS_RATE,
        max_hire_range=max_hire_range,
        max_fire_range=max_fire_range,
        best_agent_plan=best_agent_plan,
        dp_func_ref=dp_func
    )

    # Return best plan
    return max_profit, best_agent_plan[1:]


# Input from prediction
monthly_advertisers = [1024, 996, 1024, 972, 943, 859, 915, 919, 1063, 1499, 1478, 1369]
monthly_budget = [84301, 86866, 86477, 86291, 86199, 84493, 86704, 84993, 87440, 86020, 85232, 86105]

# Init input
agent_init = 652
agent_salary = 77721

# Hyperparameter
max_hire_range = 150
max_fire_range = 500

start = timer()
max_profit, plan = solve_monthly_dp_with_leftover_bonus(monthly_advertisers, monthly_budget, agent_salary, agent_init, max_hire_range, max_fire_range)
end = timer()
print(f'Running time: {end - start}')
print(plan)
