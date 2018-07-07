
root = exports ? this

RadialPlacement = () ->
    values = d3.map()
    increment = 20
    radius = 200
    center = {'x': 0, 'y': 0}
    start = -120
    current = start

    radialLocation = (center, angle, radius) ->
        x = (center.x + radius * Math.cos(angle * Math.PI / 180))
        y = (center.y + radius * Math.sin(angle * Math.PI / 180))
        {'x': x, 'y': y}

    placement = (key) ->
        value = values.get(key)
        if !values.has(key)
            value = place(key)
        value

    place = (key) ->
        value = radialLocation(center, current, radius)
        values.set(key,value)
        current += increment
        value

    setKeys = (keys) ->
        values = d3.map()

        firstCircleCount = 360 / increment

        if keys.length < firstCircleCount
            increment = 360 / keys.length

        firstCircleKeys = keys.slice(0, firstCircleCount)
        firstCircleKeys.forEach (k) -> place(k)

        secondCircleKeys = keys.slice(firstCircleCount)

        radius = radius + radius / 1.8
        increment = 360 / secondCircleKeys.length

        secondCircleKeys.forEach (k) -> place(k)

    placement.keys = (_) ->
        if !arguments.length
            return d3.keys(values)
        setKeys(_)
        placement

    placement.center = (_) ->
        if !arguments.length
            return center
        center = _
        placement

    placement.radius = (_) ->
        if !arguments.length
          return radius
        radius = _
        placement

    placement.start = (_) ->
        if !arguments.length
            return start
        start = _
        current = start
        placement

    placement.increment = (_) ->
        if !arguments.length
            return increment
        increment = _
        placement

    return placement

Network = () ->
    width = 960
    height = 800

    allData = []
    curLinksData = []
    curNodesData = []
    linkedByIndex = {}

    nodesG = null
    linksG = null

    node = null
    link = null

    layout = "force"
    filter = []
    sort = "score"

    groupCenters = null

    force = d3.layout.force()
    nodeColors = d3.scale.category20c()
    tooltip = Tooltip("vis-tooltip", 520)

    charge = (node) -> -Math.pow(node.radius, 2.0) / 2

    network = (selection, data) ->
        allData = setupData(data)

        vis = d3.select(selection).append("svg")
            .attr("width", width)
            .attr("height", height)
        linksG = vis.append("g").attr("id", "links")
        nodesG = vis.append("g").attr("id", "nodes")

        force.size([width, height])

        setLayout("force")
        setFilter("cuckoo")
        setFilter("objdump")
        setFilter("peinfo")
        setFilter("richheader")
        setFilter("label")

        update()

    update = () ->
        curNodesData = filterNodes(allData.nodes)
        curLinksData = filterLinks(allData.links, curNodesData)

        if layout == "radial"
            samples = sortedSamples(curNodesData, curLinksData)
            updateCenters(samples)

        force.nodes(curNodesData)

        updateNodes()

        if layout == "force"
            force.links(curLinksData)
            updateLinks()
        else
            force.links([])
            if link
                link.data([]).exit().remove()
                link = null

        force.start()

    network.toggleLayout = (newLayout) ->
        force.stop()
        setLayout(newLayout)
        update()

    network.toggleFilter = (toggleFilter) ->
        force.stop()
        setFilter(toggleFilter)
        update()

    network.toggleSort = (newSort) ->
        force.stop()
        setSort(newSort)
        update()

    network.updateSearch = (searchTerm) ->
        searchRegEx = new RegExp(searchTerm.toLowerCase())
        node.each (d) ->
            element = d3.select(this)
            match = d.name.toLowerCase().search(searchRegEx)
            if searchTerm.length > 0 and match >= 0
                element.style("fill", "#F38630")
                    .style("stroke-width", 2.0)
                    .style("stroke", "#555")
                d.searched = true
            else
                d.searched = false
                element.style("fill", (d) -> nodeColors(d.match))
                    .style("stroke-width", 1.0)

    network.updateData = (newData) ->
        allData = setupData(newData)
        link.remove()
        node.remove()
        update()

    setupData = (data) ->
        countExtent = d3.extent(data.nodes, (d) -> (1 - d.match) * 10)
        circleRadius = d3.scale.sqrt().range([3, 12]).domain(countExtent)

        data.nodes.forEach (n) ->
            n.x = randomnumber=Math.floor(Math.random() * width)
            n.y = randomnumber=Math.floor(Math.random() * height)
            n.radius = circleRadius(10)

        nodesMap  = mapNodes(data.nodes)

        data.links.forEach (l) ->
            l.source = nodesMap.get(l.source)
            l.target = nodesMap.get(l.target)

            linkedByIndex["#{l.source.id},#{l.target.id}"] = 1

        data

    mapNodes = (nodes) ->
        nodesMap = d3.map()
        nodes.forEach (n) ->
            nodesMap.set(n.id, n)
        nodesMap

    nodeCounts = (nodes, attr) ->
        counts = {}
        nodes.forEach (d) ->
            counts[d[attr]] ?= 0
            counts[d[attr]] += 1
        counts

    neighboring = (a, b) ->
        linkedByIndex[a.id + "," + b.id] or
            linkedByIndex[b.id + "," + a.id]

    filterNodes = (allNodes) ->
        filteredNodes = allNodes

  # Filter features here

        filteredNodes

    sortedSamples = (nodes,links) ->
        samples = []
        if sort == "score"
          counts = {}
          links.forEach (l) ->
              counts[l.source.name] ?= 0
              counts[l.source.name] += 1
              counts[l.target.name] ?= 0
              counts[l.target.name] += 1
          nodes.forEach (n) ->
              counts[n.name] ?= 0

            samples = d3.entries(counts).sort (a,b) ->
                b.value - a.value
            samples = samples.map (v) -> v.key
        else
            counts = nodeCounts(nodes, "match")
            samples = d3.entries(counts).sort (a,b) ->
                b.value - a.value
            samples = samples.map (v) -> v.key

        samples

    updateCenters = (samples) ->
        if layout == "radial"
            groupCenters = RadialPlacement().center({"x":width/2, "y":height / 2 - 100})
                .radius(300).increment(18).keys(samples)

    filterLinks = (allLinks, curNodes) ->
        curNodes = mapNodes(curNodes)
        allLinks.filter (l) ->
            curNodes.get(l.source.id) and curNodes.get(l.target.id)

    updateNodes = () ->
        node = nodesG.selectAll("circle.node")
            .data(curNodesData, (d) -> d.id)

        node.enter().append("circle")
            .attr("class", "node")
            .attr("cx", (d) -> d.x)
            .attr("cy", (d) -> d.y)
            .attr("r", (d) -> d.radius)
            .style("fill", (d) -> nodeColors(d.match))
            .style("stroke", (d) -> strokeFor(d))
            .style("stroke-width", 1.0)

        node.on("mouseover", showDetails)
            .on("mouseout", hideDetails)

        node.exit().remove()

    updateLinks = () ->
        link = linksG.selectAll("line.link")
            .data(curLinksData, (d) -> "#{d.source.id}_#{d.target.id}")
        link.enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#ddd")
            .attr("stroke-opacity", 0.8)
            .attr("x1", (d) -> d.source.x)
            .attr("y1", (d) -> d.source.y)
            .attr("x2", (d) -> d.target.x)
            .attr("y2", (d) -> d.target.y)

        link.exit().remove()

    setLayout = (newLayout) ->
        layout = newLayout
        if layout == "force"
            force.on("tick", forceTick)
                .charge(-200)
                .linkDistance(100)
        else if layout == "radial"
            force.on("tick", radialTick)
                .charge(charge)

    setFilter = (toggleFilter) ->
        if toggleFilter in filter
            filter.splice(filter.indexOf(toggleFilter), 1)
        else
            filter.push toggleFilter

        filter

    setSort = (newSort) ->
        sort = newSort

    forceTick = (e) ->
        node
            .attr("cx", (d) -> d.x)
            .attr("cy", (d) -> d.y)

        link
            .attr("x1", (d) -> d.source.x)
            .attr("y1", (d) -> d.source.y)
            .attr("x2", (d) -> d.target.x)
            .attr("y2", (d) -> d.target.y)

    radialTick = (e) ->
        node.each(moveToRadialLayout(e.alpha))

        node
            .attr("cx", (d) -> d.x)
            .attr("cy", (d) -> d.y)

        if e.alpha < 0.03
            force.stop()
            updateLinks()

    moveToRadialLayout = (alpha) ->
        k = alpha * 0.1
        (d) ->
            centerNode = groupCenters(d.artist)
            d.x += (centerNode.x - d.x) * k
            d.y += (centerNode.y - d.y) * k


    strokeFor = (d) ->
        d3.rgb(nodeColors(d.match)).darker().toString()

    showDetails = (d,i) ->
        content = '<p class="main">' + d.name + '</span></p>'
        content += '<hr class="tooltip-hr">'
        content += '<p class="main">' + d.artist + '</span></p>'
        tooltip.showTooltip(content,d3.event)

        if link
            link.attr("stroke", (l) ->
              if l.source == d or l.target == d then "#555" else "#ddd"
            )
                .attr("stroke-opacity", (l) ->
                    if l.source == d or l.target == d then 1.0 else 0.5
                )

        node.style("stroke", (n) ->
            if (n.searched or neighboring(d, n)) then "#555" else strokeFor(n))
            .style("stroke-width", (n) ->
                if (n.searched or neighboring(d, n)) then 2.0 else 1.0)

        d3.select(this).style("stroke","black")
            .style("stroke-width", 2.0)

    hideDetails = (d,i) ->
        tooltip.hideTooltip()
        node.style("stroke", (n) -> if !n.searched then strokeFor(n) else "#555")
            .style("stroke-width", (n) -> if !n.searched then 1.0 else 2.0)
        if link
            link.attr("stroke", "#ddd")
                .attr("stroke-opacity", 0.8)

    return network

activate = (group, link) ->
    d3.selectAll("##{group} a").classed("active", false)
    d3.select("##{group} ##{link}").classed("active", true)

activateFilters = (group, link) ->
    d3.select("##{group} ##{link}").classed("active", !d3.select("##{group} ##{link}").classed("active"))


$ ->
    malwareNetwork = Network()

    d3.selectAll("#layouts a").on "click", (d) ->
        newLayout = d3.select(this).attr("id")
        activate("layouts", newLayout)
        malwareNetwork.toggleLayout(newLayout)

    d3.selectAll("#filters a").on "click", (d) ->
        toggleFilter = d3.select(this).attr("id")
        activateFilters("filters", toggleFilter)
        malwareNetwork.toggleFilter(toggleFilter)

    d3.selectAll("#sorts a").on "click", (d) ->
        newSort = d3.select(this).attr("id")
        activate("sorts", newSort)
        malwareNetwork.toggleSort(newSort)

    $("#sample_search").on "change", (e) ->
        sampleSha256 = $(this).val()
        if sampleSha256.length == 64
            relationship_query = new proto.feedhandling.Query();
            relationship_query.setSha256(sampleSha256);

            nodes = []
            links = []

            stream = service.queryRelationship(relationship_query, {});
            stream.on "data", (response) ->
                node =
                    id: response.getSha256()
                    name: response.getSha256()
                    artist: response.getLabelsList().join()
                    match: response.getDistance()
                nodes.push node

                if sampleSha256 != response.getSha256()
                    link =
                        source: sampleSha256
                        target: response.getSha256()
                    links.push link

                if nodes.length == 25
                    relationships =
                        nodes: nodes
                        links: links
                    malwareNetwork.updateData(relationships)

    $("#search").keyup () ->
        searchTerm = $(this).val()
        malwareNetwork.updateSearch(searchTerm)

    d3.json "data/sample.json", (json) ->
        malwareNetwork("#vis", json)
