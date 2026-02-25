%% description
% One-shot RTD script for the TurtleBot. Randomly generates a single
% obstacle, loads the appropriate FRS for the chosen initial speed, runs
% one trajectory optimization, and plots the result.
%
% Based on step_4_ex_2_trajectory_optimization.m
%
%% user parameters
v_0 = 0.5 ; % initial speed (m/s), must be in [0.0, 1.5]

% robot desired location
x_des = 0.75 ;
y_des = 0.5 ;

% obstacle
N_vertices = 5 ;
obstacle_scale = 0.3 ;
obstacle_buffer = 0.05 ; % m

%% automated from here
% load FRS based on initial speed
disp('Loading FRS...')
if v_0 >= 1.0 && v_0 <= 1.5
    FRS = load('turtlebot_FRS_deg_10_v_0_1.0_to_1.5.mat') ;
elseif v_0 >= 0.5
    FRS = load('turtlebot_FRS_deg_10_v_0_0.5_to_1.0.mat') ;
elseif v_0 >= 0.0
    FRS = load('turtlebot_FRS_deg_10_v_0_0.0_to_0.5.mat') ;
else
    error('Please pick an initial speed between 0.0 and 1.5 m/s')
end

% create turtlebot
A = turtlebot_agent ;
A.reset([0;0;0;v_0])

% randomly place obstacle somewhere in front of the robot
obs_x = 0.5 + rand()*0.75 ;   % random x in [0.5, 1.25]
obs_y = (rand() - 0.5) * 0.8 ; % random y in [-0.4, 0.4]
obstacle_location = [obs_x ; obs_y] ;
O = make_random_polygon(N_vertices, obstacle_location, obstacle_scale) ;

%% create cost function
z_goal = [x_des; y_des] ;
z_goal_local = world_to_local(A.state(:,end), z_goal) ;
cost = @(k) turtlebot_cost_for_fmincon(k, FRS, z_goal_local) ;

%% create constraint function
point_spacing = compute_turtlebot_point_spacings(A.footprint, obstacle_buffer) ;
[O_FRS, O_buf, O_pts] = compute_turtlebot_discretized_obs(O, ...
                    A.state(:,end), obstacle_buffer, point_spacing, FRS) ;

FRS_msspoly = FRS.FRS_polynomial - 1 ;
k = FRS.k ;
z = FRS.z ;

FRS_poly = get_FRS_polynomial_structure(FRS_msspoly, z, k) ;
FRS_poly_viz = subs(FRS_msspoly, k, [k(2);k(1)]) ;

cons_poly = evaluate_FRS_polynomial_on_obstacle_points(FRS_poly, O_FRS) ;
cons_poly_grad = get_constraint_polynomial_gradient(cons_poly) ;
nonlcon = @(k) turtlebot_nonlcon_for_fmincon(k, cons_poly, cons_poly_grad) ;

% bounds on yaw rate parameter
k_1_bounds = [-1, 1] ;

% bounds on speed parameter
v_max = FRS.v_range(2) ;
v_des_lo = max(v_0 - FRS.delta_v, FRS.v_range(1)) ;
v_des_hi = min(v_0 + FRS.delta_v, FRS.v_range(2)) ;
k_2_lo = (v_des_lo - v_max/2)*(2/v_max) ;
k_2_hi = (v_des_hi - v_max/2)*(2/v_max) ;
k_2_bounds = [k_2_lo, k_2_hi] ;

k_bounds = [k_1_bounds ; k_2_bounds] ;

%% run trajectory optimization
initial_guess = zeros(2,1) ;

options = optimoptions('fmincon', ...
    'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 1e5, ...
    'OptimalityTolerance', 1e-3, ...
    'CheckGradients', false, ...
    'FiniteDifferenceType', 'central', ...
    'Diagnostics', 'off', ...
    'SpecifyConstraintGradient', true, ...
    'SpecifyObjectiveGradient', true) ;

[k_opt, ~, exitflag] = fmincon(cost, ...
                               initial_guess, ...
                               [], [], [], [], ...
                               k_bounds(:,1), k_bounds(:,2), ...
                               nonlcon, options) ;

if exitflag <= 0
    disp('No feasible trajectory found!')
    k_opt = [] ;
else
    disp('Feasible trajectory found.')
end

%% get FRS contour for optimal k
if ~isempty(k_opt)
    I_z_opt = msubs(FRS_msspoly, k, k_opt) ;
    x0 = FRS.initial_x ;
    y0 = FRS.initial_y ;
    D  = FRS.distance_scale ;
    C_FRS   = get_2D_contour_points(I_z_opt, z, 0) ;
    C_world = FRS_to_world(C_FRS, A.state(:,end), x0, y0, D) ;
end

%% move robot along the safe trajectory
if ~isempty(k_opt)
    w_des = full(msubs(FRS.w_des, k, k_opt)) ;
    v_des = full(msubs(FRS.v_des, k, k_opt)) ;
    t_plan = FRS.t_plan ;
    t_stop = v_des / A.max_accel ;
    [T_brk, U_brk, Z_brk] = make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des) ;
    A.move(T_brk(end), T_brk, U_brk, Z_brk) ;
end

%% get parameter-space obstacle contours
I_k = msubs(FRS_poly_viz, z, O_FRS) ;

%% plot
figure(1) ; clf ;

% --- world frame ---
subplot(1,3,3) ; hold on ; axis equal ; set(gca,'FontSize',15)
plot(A)
patch(O_buf(1,:), O_buf(2,:), [1 0.5 0.6])
patch(O(1,:),     O(2,:),     [1 0.7 0.8])
plot(O_pts(1,:),  O_pts(2,:), '.', 'Color', [0.5 0.1 0.1], 'MarkerSize', 15)
plot(x_des, y_des, 'k*', 'LineWidth', 2, 'MarkerSize', 15)
if ~isempty(k_opt)
    plot_path(Z_brk, 'b--', 'LineWidth', 1.5)
    plot(C_world(1,:), C_world(2,:), 'Color', [0.3 0.8 0.5], 'LineWidth', 1.5)
end
axis([-0.5, 1.5, -1, 1])
title('World Frame') ; xlabel('x [m]') ; ylabel('y [m]')

% --- FRS frame ---
subplot(1,3,2) ; hold on ; axis equal ; grid on ; set(gca,'FontSize',15)
plot_2D_msspoly_contour(FRS.h_Z0, z, 0, 'Color', [0 0 1], 'LineWidth', 1.5)
plot(O_FRS(1,:), O_FRS(2,:), '.', 'Color', [0.5 0.1 0.1], 'MarkerSize', 15)
if ~isempty(k_opt)
    plot(C_FRS(1,:), C_FRS(2,:), 'Color', [0.3 0.8 0.5], 'LineWidth', 1.5)
end
title('FRS Frame') ; xlabel('x (scaled)') ; ylabel('y (scaled)')

% --- trajectory parameter space ---
subplot(1,3,1) ; hold on ; axis equal ; set(gca,'FontSize',15)
for idx = 1:length(I_k)
    plot_2D_msspoly_contour(I_k(idx), k, 0, 'FillColor', [1 0.5 0.6])
end
if ~isempty(k_opt)
    plot(k_opt(2), k_opt(1), '.', 'Color', [0.3 0.8 0.5], 'MarkerSize', 15)
    plot(k_opt(2), k_opt(1), 'ko', 'MarkerSize', 6)
end
title('Traj Params') ; xlabel('speed param') ; ylabel('yaw rate param')

% --- figure 2: world frame only ---
figure(2) ; clf ; hold on ; axis equal ; set(gca,'FontSize',15)
plot(A)
patch(O_buf(1,:), O_buf(2,:), [1 0.5 0.6])
patch(O(1,:),     O(2,:),     [1 0.7 0.8])
plot(O_pts(1,:),  O_pts(2,:), '.', 'Color', [0.5 0.1 0.1], 'MarkerSize', 15)
plot(x_des, y_des, 'k*', 'LineWidth', 2, 'MarkerSize', 15)
if ~isempty(k_opt)
    plot_path(Z_brk, 'b--', 'LineWidth', 1.5)
    plot(C_world(1,:), C_world(2,:), 'Color', [0.3 0.8 0.5], 'LineWidth', 1.5)
end
axis([-0.5, 1.5, -1, 1])
title('World Frame') ; xlabel('x [m]') ; ylabel('y [m]')
