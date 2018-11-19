function sysCall_init()
    drone_handle = sim.getObjectHandle('eDroneBase')
    collection_handles= sim.getCollectionHandle('Obstacles')

    obstacles_handles = {}
    for i=1,6 do
        table.insert(obstacles_handles,sim.getObjectHandle('obstacle_'..tostring(i)))
    end

    start_handle = sim.getObjectHandle('DroneDummy')
    goal_handle = sim.getObjectHandle('GoalDummy')
    collection_handles= sim.getCollectionHandle('Obstacles')

    t=simOMPL.createTask('t')
    ss={simOMPL.createStateSpace('6d',simOMPL.StateSpaceType.pose3d,start_handle,{-1.0,-1.4,0},{2,6,4},1)}
    simOMPL.setStateSpace(t,ss)
    simOMPL.setAlgorithm(t,simOMPL.Algorithm.RRTConnect)
    simOMPL.setCollisionPairs(t,{sim.getObjectHandle('eDroneVisible'),collection_handles})

    no_of_path_points_required = 50
    compute_path_flag = false

    path_pub=simROS.advertise('/vrep/waypoints', 'geometry_msgs/PoseArray')
    whycon_sub = simROS.subscribe('/path_target', 'geometry_msgs/Pose', 'activate_path')
end

function activate_path(msg)
    compute_path_flag = true
    sim.setObjectPosition(goal_handle, -1, { msg.position.x, msg.position.y, msg.position.z})
end

function visualizePath( path )
    if not _lineContainer then
        _lineContainer=sim.addDrawingObject(sim.drawing_lines,1,0,-1,99999,{0.2,0.2,1})
    end
    sim.addDrawingObjectItem(_lineContainer,nil)
    if path then
        local pc=#path/7
        for i=1,pc-1,1 do
            lineDat={path[(i-1)*7+1],path[(i-1)*7+2],path[(i-1)*7+3],path[i*7+1],path[i*7+2],path[i*7+3]}
            sim.addDrawingObjectItem(_lineContainer,lineDat)
        end
    end
end

function packdata(path)

    local sender = {header = {}, poses = {}}
    
    sender['header']={seq=123, stamp=simROS.getTime(), frame_id="drone"}
    sender['poses'] = {}

    for i=1,#path,7 do
        a = {x = 0, y = 0, w = 0, z = 0}
        b = {x = 0, y = 0, z = 0}
        pose = {position = b, orientation = a, }
        pose.position.x = path[i]
        pose.position.y = path[i+1]
        pose.position.z = path[i+2]
        sender.poses[math.floor(i/7) + 1] = pose
    end
    return sender
end
function compute_and_send_path(task)
    local r
    local path

    r,path=simOMPL.compute(t,10,-1,no_of_path_points_required)

    if(r == true) then
        visualizePath(path)
        message = packdata(path)  
        simROS.publish(path_pub,message)
    end
    return r
end

function getpose(handle,ref_handle)
    position = sim.getObjectPosition(handle,ref_handle)
    orientation = sim.getObjectQuaternion(handle,ref_handle)
    pose = {position[1],position[2],position[3],orientation[1],orientation[2],orientation[3],orientation[4]}
    return pose
end

function sysCall_actuation()
    if compute_path_flag == true then
        start_pose = getpose(start_handle,-1)
        goal_pose = getpose(goal_handle,-1)
        simOMPL.setStartState(t,start_pose)
        simOMPL.setGoalState(t,{goal_pose[1],goal_pose[2],goal_pose[3],start_pose[4],start_pose[5],start_pose[6],start_pose[7]})
        status = compute_and_send_path(t)
        if(status == true) then -- path computed
            compute_path_flag = false
        end
    end 
end
function sysCall_cleanup()

end

