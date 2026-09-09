"""Clock Domain Crossing (CDC) Wrapper for RailNet Dual-Clock Architecture.

Provides safe asynchronous clock-domain isolation between the host interface
(PCIe/Wishbone at 50-125 MHz) and the compute core (RailNet at 200-500 MHz):

- AxisCDCFIFO:          Gray-coded AsyncFIFO wrapper for AXI4-Stream crossing.
- AxiLiteCDCBridge:     Request/Response AsyncFIFO bridge for AXI4-Lite CSR.
- RailNetDualClockTop:  Complete dual-clock accelerator top-level.

The single-clock RailNetTop remains unmodified — this module wraps it via
DomainRenamer(sync→core) and adds CDC primitives on all domain boundaries.
"""

from amaranth import (
    Cat, ClockDomain, ClockSignal, Const, DomainRenamer,
    Module, Mux, ResetSignal, Signal, signed,
)
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.lib.fifo import AsyncFIFO
from amaranth.lib.cdc import FFSynchronizer, PulseSynchronizer

from hardware.rtl.top import RailNetTop
from hardware.rtl.int8_tile import ACC_W, RAILS_DEFAULT


# ---------------------------------------------------------------------------
#  1.  AXI4-Stream CDC FIFO
# ---------------------------------------------------------------------------


class AxisCDCFIFO(wiring.Component):
    """AXI4-Stream Clock Domain Crossing via Gray-coded AsyncFIFO.

    Crosses an AXI4-Stream channel from *w_domain* to *r_domain* using
    Amaranth's ``AsyncFIFO`` with Gray-coded pointers (metastability-safe).

    The FIFO word packs ``{tlast, tdata}`` into a single entry so that
    ``tlast`` is atomically transferred together with the last data beat.

    Parameters
    ----------
    data_width : int
        Width of ``tdata`` in bits.
    depth : int
        FIFO depth (must be a power of two).  Default **32**.
    w_domain : str
        Producer clock domain name.
    r_domain : str
        Consumer clock domain name.
    """

    def __init__(self, data_width: int = 32, depth: int = 32,
                 w_domain: str = "host", r_domain: str = "core"):
        self.data_width = data_width
        self.depth = depth
        self.w_domain = w_domain
        self.r_domain = r_domain

        super().__init__({
            # Producer (w_domain) side
            "w_tdata":  In(data_width),
            "w_tvalid": In(1),
            "w_tready": Out(1),
            "w_tlast":  In(1),
            # Consumer (r_domain) side
            "r_tdata":  Out(data_width),
            "r_tvalid": Out(1),
            "r_tready": In(1),
            "r_tlast":  Out(1),
        })

    def elaborate(self, platform):
        m = Module()

        fifo = AsyncFIFO(
            width=self.data_width + 1,       # +1 for tlast
            depth=self.depth,
            w_domain=self.w_domain,
            r_domain=self.r_domain,
        )
        m.submodules.fifo = fifo

        # Write side — pack {tlast, tdata}
        m.d.comb += [
            fifo.w_data.eq(Cat(self.w_tdata, self.w_tlast)),
            fifo.w_en.eq(self.w_tvalid & fifo.w_rdy),
            self.w_tready.eq(fifo.w_rdy),
        ]

        # Read side — unpack {tlast, tdata}
        m.d.comb += [
            self.r_tdata.eq(fifo.r_data[:self.data_width]),
            self.r_tlast.eq(fifo.r_data[self.data_width]),
            self.r_tvalid.eq(fifo.r_rdy),
            fifo.r_en.eq(self.r_tready & fifo.r_rdy),
        ]

        return m


# ---------------------------------------------------------------------------
#  2.  AXI4-Lite CDC Bridge (Request / Response FIFO pair)
# ---------------------------------------------------------------------------


class AxiLiteCDCBridge(wiring.Component):
    """AXI4-Lite Clock Domain Crossing via paired AsyncFIFOs.

    Host side (AXI-Lite **slave**) lives in ``host_domain``.
    Core side (AXI-Lite **master**) lives in ``core_domain``.

    Request  FIFO (host→core): ``{is_write, addr, wdata, wstrb}`` = 49 bits.
    Response FIFO (core→host): ``{rdata, resp}``                  = 34 bits.
    """

    def __init__(self, addr_w: int = 12, data_w: int = 32,
                 host_domain: str = "host", core_domain: str = "core"):
        self.addr_w = addr_w
        self.data_w = data_w
        self.host_domain = host_domain
        self.core_domain = core_domain

        super().__init__({
            # ---------- Host-side AXI-Lite Slave ----------
            "h_awaddr":  In(addr_w),  "h_awvalid": In(1),  "h_awready": Out(1),
            "h_wdata":   In(data_w),  "h_wstrb":   In(4),
            "h_wvalid":  In(1),       "h_wready":  Out(1),
            "h_bresp":   Out(2),      "h_bvalid":  Out(1), "h_bready": In(1),
            "h_araddr":  In(addr_w),  "h_arvalid": In(1),  "h_arready": Out(1),
            "h_rdata":   Out(data_w), "h_rresp":   Out(2),
            "h_rvalid":  Out(1),      "h_rready":  In(1),
            # ---------- Core-side AXI-Lite Master ----------
            "c_awaddr":  Out(addr_w), "c_awvalid": Out(1), "c_awready": In(1),
            "c_wdata":   Out(data_w), "c_wstrb":   Out(4),
            "c_wvalid":  Out(1),      "c_wready":  In(1),
            "c_bresp":   In(2),       "c_bvalid":  In(1),  "c_bready": Out(1),
            "c_araddr":  Out(addr_w), "c_arvalid": Out(1), "c_arready": In(1),
            "c_rdata":   In(data_w),  "c_rresp":   In(2),
            "c_rvalid":  In(1),       "c_rready":  Out(1),
        })

    def elaborate(self, platform):
        m = Module()
        hd = self.host_domain
        cd = self.core_domain

        # Request FIFO: 1 + 12 + 32 + 4 = 49 bits, depth 4
        REQ_W = 1 + self.addr_w + self.data_w + 4
        req_fifo = AsyncFIFO(width=REQ_W, depth=4, w_domain=hd, r_domain=cd)
        m.submodules.req_fifo = req_fifo

        # Response FIFO: 2 + 32 = 34 bits, depth 4
        RSP_W = 2 + self.data_w
        rsp_fifo = AsyncFIFO(width=RSP_W, depth=4, w_domain=cd, r_domain=hd)
        m.submodules.rsp_fifo = rsp_fifo

        # ----------------------------------------------------------------
        #  HOST-SIDE FSM  (clocked by host_domain)
        # ----------------------------------------------------------------
        H_IDLE       = 0
        H_CAPTURE_W  = 1
        H_PUSH_WR    = 2
        H_WAIT_WR    = 3
        H_PUSH_RD    = 4
        H_WAIT_RD    = 5

        h_state   = Signal(range(6), reset=H_IDLE)
        h_addr    = Signal(self.addr_w)
        h_wdata_r = Signal(self.data_w)
        h_wstrb_r = Signal(4)

        # Default combinational outputs
        m.d.comb += [
            self.h_awready.eq(0), self.h_wready.eq(0), self.h_arready.eq(0),
            self.h_bvalid.eq(0),  self.h_bresp.eq(0),
            self.h_rvalid.eq(0),  self.h_rresp.eq(0),  self.h_rdata.eq(0),
            req_fifo.w_en.eq(0),  req_fifo.w_data.eq(0),
            rsp_fifo.r_en.eq(0),
        ]

        with m.Switch(h_state):
            with m.Case(H_IDLE):
                m.d.comb += [
                    self.h_awready.eq(1),
                    self.h_wready.eq(1),
                    self.h_arready.eq(~self.h_awvalid),
                ]
                with m.If(self.h_awvalid & self.h_wvalid):
                    m.d[hd] += [
                        h_addr.eq(self.h_awaddr), h_wdata_r.eq(self.h_wdata),
                        h_wstrb_r.eq(self.h_wstrb), h_state.eq(H_PUSH_WR),
                    ]
                with m.Elif(self.h_awvalid):
                    m.d[hd] += [h_addr.eq(self.h_awaddr), h_state.eq(H_CAPTURE_W)]
                with m.Elif(self.h_arvalid):
                    m.d[hd] += [h_addr.eq(self.h_araddr), h_state.eq(H_PUSH_RD)]

            with m.Case(H_CAPTURE_W):
                m.d.comb += self.h_wready.eq(1)
                with m.If(self.h_wvalid):
                    m.d[hd] += [
                        h_wdata_r.eq(self.h_wdata), h_wstrb_r.eq(self.h_wstrb),
                        h_state.eq(H_PUSH_WR),
                    ]

            with m.Case(H_PUSH_WR):
                m.d.comb += [
                    req_fifo.w_data.eq(Cat(Const(1, 1), h_addr, h_wdata_r, h_wstrb_r)),
                    req_fifo.w_en.eq(1),
                ]
                with m.If(req_fifo.w_rdy):
                    m.d[hd] += h_state.eq(H_WAIT_WR)

            with m.Case(H_WAIT_WR):
                with m.If(rsp_fifo.r_rdy):
                    m.d.comb += [
                        self.h_bvalid.eq(1),
                        self.h_bresp.eq(rsp_fifo.r_data[:2]),
                    ]
                    with m.If(self.h_bready):
                        m.d.comb += rsp_fifo.r_en.eq(1)
                        m.d[hd] += h_state.eq(H_IDLE)

            with m.Case(H_PUSH_RD):
                m.d.comb += [
                    req_fifo.w_data.eq(Cat(Const(0, 1), h_addr,
                                           Const(0, self.data_w + 4))),
                    req_fifo.w_en.eq(1),
                ]
                with m.If(req_fifo.w_rdy):
                    m.d[hd] += h_state.eq(H_WAIT_RD)

            with m.Case(H_WAIT_RD):
                with m.If(rsp_fifo.r_rdy):
                    m.d.comb += [
                        self.h_rvalid.eq(1),
                        self.h_rdata.eq(rsp_fifo.r_data[2:]),
                        self.h_rresp.eq(rsp_fifo.r_data[:2]),
                    ]
                    with m.If(self.h_rready):
                        m.d.comb += rsp_fifo.r_en.eq(1)
                        m.d[hd] += h_state.eq(H_IDLE)

        # ----------------------------------------------------------------
        #  CORE-SIDE FSM  (clocked by core_domain)
        # ----------------------------------------------------------------
        C_IDLE       = 0
        C_WR_ADDR    = 1
        C_WR_DATA    = 2
        C_WR_RSP     = 3
        C_RD_ADDR    = 4
        C_RD_DATA    = 5
        C_PUSH_RSP   = 6

        c_state   = Signal(range(7), reset=C_IDLE)
        c_addr    = Signal(self.addr_w)
        c_wdata   = Signal(self.data_w)
        c_wstrb   = Signal(4)
        c_rdata   = Signal(self.data_w)
        c_resp    = Signal(2)

        m.d.comb += [
            self.c_awaddr.eq(c_addr),  self.c_awvalid.eq(0),
            self.c_wdata.eq(c_wdata),  self.c_wstrb.eq(c_wstrb),
            self.c_wvalid.eq(0),       self.c_bready.eq(0),
            self.c_araddr.eq(c_addr),  self.c_arvalid.eq(0),
            self.c_rready.eq(0),
            req_fifo.r_en.eq(0),
            rsp_fifo.w_en.eq(0), rsp_fifo.w_data.eq(0),
        ]

        with m.Switch(c_state):
            with m.Case(C_IDLE):
                with m.If(req_fifo.r_rdy):
                    m.d.comb += req_fifo.r_en.eq(1)
                    a_lo = 1
                    a_hi = 1 + self.addr_w
                    d_hi = a_hi + self.data_w
                    s_hi = d_hi + 4
                    m.d[cd] += [
                        c_addr.eq(req_fifo.r_data[a_lo:a_hi]),
                        c_wdata.eq(req_fifo.r_data[a_hi:d_hi]),
                        c_wstrb.eq(req_fifo.r_data[d_hi:s_hi]),
                    ]
                    with m.If(req_fifo.r_data[0]):          # is_write
                        m.d[cd] += c_state.eq(C_WR_ADDR)
                    with m.Else():
                        m.d[cd] += c_state.eq(C_RD_ADDR)

            with m.Case(C_WR_ADDR):
                m.d.comb += [self.c_awvalid.eq(1), self.c_wvalid.eq(1)]
                with m.If(self.c_awready & self.c_wready):
                    m.d[cd] += c_state.eq(C_WR_RSP)
                with m.Elif(self.c_awready):
                    m.d[cd] += c_state.eq(C_WR_DATA)

            with m.Case(C_WR_DATA):
                m.d.comb += self.c_wvalid.eq(1)
                with m.If(self.c_wready):
                    m.d[cd] += c_state.eq(C_WR_RSP)

            with m.Case(C_WR_RSP):
                m.d.comb += self.c_bready.eq(1)
                with m.If(self.c_bvalid):
                    m.d[cd] += [c_resp.eq(self.c_bresp), c_state.eq(C_PUSH_RSP)]

            with m.Case(C_RD_ADDR):
                m.d.comb += self.c_arvalid.eq(1)
                with m.If(self.c_arready):
                    m.d[cd] += c_state.eq(C_RD_DATA)

            with m.Case(C_RD_DATA):
                m.d.comb += self.c_rready.eq(1)
                with m.If(self.c_rvalid):
                    m.d[cd] += [
                        c_rdata.eq(self.c_rdata), c_resp.eq(self.c_rresp),
                        c_state.eq(C_PUSH_RSP),
                    ]

            with m.Case(C_PUSH_RSP):
                m.d.comb += [
                    rsp_fifo.w_data.eq(Cat(c_resp, c_rdata)),
                    rsp_fifo.w_en.eq(1),
                ]
                with m.If(rsp_fifo.w_rdy):
                    m.d[cd] += c_state.eq(C_IDLE)

        return m


# ---------------------------------------------------------------------------
#  3.  RailNet Dual-Clock Top  (host_clk + core_clk)
# ---------------------------------------------------------------------------


class RailNetDualClockTop(wiring.Component):
    """Dual-clock RailNet AI Accelerator top-level.

    Wraps the existing single-clock ``RailNetTop`` (unchanged) with full
    asynchronous CDC isolation:

    * **Host domain** (``host_clk``): AXI-Lite CSR, AXI-Stream external
      interface, IRQ, programming pins.
    * **Core domain** (``core_clk``): Entire compute pipeline — Grid,
      Broadcaster, Gather, FSM, CSR internal registers.

    Data path crossings use 32-deep Gray-coded ``AsyncFIFO``'s.
    Control crossings use ``PulseSynchronizer`` (strobes) and
    ``FFSynchronizer`` (levels).

    When ``core_clk == host_clk`` (single-clock fallback), the design
    degenerates to exactly the same behaviour as ``RailNetTop``.
    """

    def __init__(
        self,
        num_tiles: int = 4,
        rails: int = RAILS_DEFAULT,
        codebook_depth: int = 64,
        route_depth: int = 512,
        fifo_depth: int = 32,
    ):
        self.num_tiles = num_tiles
        self.rails = rails
        self.codebook_depth = codebook_depth
        self.route_depth = route_depth
        self.fifo_depth = fifo_depth
        self.tidx_w = max(1, (num_tiles - 1).bit_length())

        super().__init__(
            {
                # ---- AXI4-Lite Slave (host domain) ----
                "s_axi_awaddr":  In(12),  "s_axi_awvalid": In(1),
                "s_axi_awready": Out(1),
                "s_axi_wdata":   In(32),  "s_axi_wstrb": In(4),
                "s_axi_wvalid":  In(1),   "s_axi_wready": Out(1),
                "s_axi_bresp":   Out(2),  "s_axi_bvalid": Out(1),
                "s_axi_bready":  In(1),
                "s_axi_araddr":  In(12),  "s_axi_arvalid": In(1),
                "s_axi_arready": Out(1),
                "s_axi_rdata":   Out(32), "s_axi_rresp": Out(2),
                "s_axi_rvalid":  Out(1),  "s_axi_rready": In(1),
                "irq": Out(1),
                # ---- AXI4-Stream Slave (host domain) ----
                "s_axis_tdata":  In(32),  "s_axis_tvalid": In(1),
                "s_axis_tready": Out(1),  "s_axis_tlast":  In(1),
                # ---- AXI4-Stream Master (host domain) ----
                "m_axis_tdata":  Out(signed(ACC_W)),
                "m_axis_tvalid": Out(1),  "m_axis_tready": In(1),
                "m_axis_tlast":  Out(1),
                # ---- Programming (host domain — sync'd internally) ----
                "prog_tile_idx":   In(self.tidx_w),
                "prog_route_en":   In(1),
                "prog_route_addr": In(max(1, (route_depth - 1).bit_length())),
                "prog_route_data": In(16),
                "prog_cb_en":      In(1),
                "prog_cb_addr":    In(16),
                "prog_cb_data":    In(27),
                "prog_rail_en":    In(1),
                "prog_rail_addr":  In(max(1, (rails - 1).bit_length())),
                "prog_rail_data":  In(signed(8)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        # ---- Create clock domains ----
        # Clock signals are driven externally: by the simulator (add_clock),
        # by Verilog wrapper ports, or by the Caravel PLL harness.
        m.domains += ClockDomain("host")
        m.domains += ClockDomain("core")

        # ---- Inner RailNetTop — entirely in core domain ----
        inner = DomainRenamer({"sync": "core"})(
            RailNetTop(
                num_tiles=self.num_tiles,
                rails=self.rails,
                codebook_depth=self.codebook_depth,
                route_depth=self.route_depth,
            )
        )
        m.submodules.inner = inner

        # ---- AXI-Lite CDC Bridge (host ↔ core) ----
        axi_cdc = AxiLiteCDCBridge(host_domain="host", core_domain="core")
        m.submodules.axi_cdc = axi_cdc

        # Host-side: external AXI-Lite ports → bridge host slave
        m.d.comb += [
            axi_cdc.h_awaddr.eq(self.s_axi_awaddr),
            axi_cdc.h_awvalid.eq(self.s_axi_awvalid),
            self.s_axi_awready.eq(axi_cdc.h_awready),
            axi_cdc.h_wdata.eq(self.s_axi_wdata),
            axi_cdc.h_wstrb.eq(self.s_axi_wstrb),
            axi_cdc.h_wvalid.eq(self.s_axi_wvalid),
            self.s_axi_wready.eq(axi_cdc.h_wready),
            self.s_axi_bresp.eq(axi_cdc.h_bresp),
            self.s_axi_bvalid.eq(axi_cdc.h_bvalid),
            axi_cdc.h_bready.eq(self.s_axi_bready),
            axi_cdc.h_araddr.eq(self.s_axi_araddr),
            axi_cdc.h_arvalid.eq(self.s_axi_arvalid),
            self.s_axi_arready.eq(axi_cdc.h_arready),
            self.s_axi_rdata.eq(axi_cdc.h_rdata),
            self.s_axi_rresp.eq(axi_cdc.h_rresp),
            self.s_axi_rvalid.eq(axi_cdc.h_rvalid),
            axi_cdc.h_rready.eq(self.s_axi_rready),
        ]

        # Core-side: bridge core master → inner RailNetTop AXI-Lite
        m.d.comb += [
            inner.s_axi_awaddr.eq(axi_cdc.c_awaddr),
            inner.s_axi_awvalid.eq(axi_cdc.c_awvalid),
            axi_cdc.c_awready.eq(inner.s_axi_awready),
            inner.s_axi_wdata.eq(axi_cdc.c_wdata),
            inner.s_axi_wstrb.eq(axi_cdc.c_wstrb),
            inner.s_axi_wvalid.eq(axi_cdc.c_wvalid),
            axi_cdc.c_wready.eq(inner.s_axi_wready),
            axi_cdc.c_bresp.eq(inner.s_axi_bresp),
            axi_cdc.c_bvalid.eq(inner.s_axi_bvalid),
            inner.s_axi_bready.eq(axi_cdc.c_bready),
            inner.s_axi_araddr.eq(axi_cdc.c_araddr),
            inner.s_axi_arvalid.eq(axi_cdc.c_arvalid),
            axi_cdc.c_arready.eq(inner.s_axi_arready),
            axi_cdc.c_rdata.eq(inner.s_axi_rdata),
            axi_cdc.c_rresp.eq(inner.s_axi_rresp),
            axi_cdc.c_rvalid.eq(inner.s_axi_rvalid),
            inner.s_axi_rready.eq(axi_cdc.c_rready),
        ]

        # ---- AXI-Stream Input CDC (host → core) ----
        in_cdc = AxisCDCFIFO(
            data_width=32, depth=self.fifo_depth,
            w_domain="host", r_domain="core",
        )
        m.submodules.in_cdc = in_cdc

        m.d.comb += [
            in_cdc.w_tdata.eq(self.s_axis_tdata),
            in_cdc.w_tvalid.eq(self.s_axis_tvalid),
            self.s_axis_tready.eq(in_cdc.w_tready),
            in_cdc.w_tlast.eq(self.s_axis_tlast),

            inner.s_axis_tdata.eq(in_cdc.r_tdata),
            inner.s_axis_tvalid.eq(in_cdc.r_tvalid),
            in_cdc.r_tready.eq(inner.s_axis_tready),
            inner.s_axis_tlast.eq(in_cdc.r_tlast),
        ]

        # ---- AXI-Stream Output CDC (core → host) ----
        out_cdc = AxisCDCFIFO(
            data_width=ACC_W, depth=self.fifo_depth,
            w_domain="core", r_domain="host",
        )
        m.submodules.out_cdc = out_cdc

        m.d.comb += [
            out_cdc.w_tdata.eq(inner.m_axis_tdata),
            out_cdc.w_tvalid.eq(inner.m_axis_tvalid),
            inner.m_axis_tready.eq(out_cdc.w_tready),
            out_cdc.w_tlast.eq(inner.m_axis_tlast),

            self.m_axis_tdata.eq(out_cdc.r_tdata),
            self.m_axis_tvalid.eq(out_cdc.r_tvalid),
            out_cdc.r_tready.eq(self.m_axis_tready),
            self.m_axis_tlast.eq(out_cdc.r_tlast),
        ]

        # ---- IRQ synchronisation (core → host) ----
        irq_sync = FFSynchronizer(i=inner.irq, o=self.irq, o_domain="host")
        m.submodules.irq_sync = irq_sync

        # ---- Programming port CDC (host → core) via Atomic AsyncFIFO ----
        # Eliminates multi-bit bus skew hazards and race conditions between strobe and data.
        # Word format: Cat(p_type[2], p_tile[tidx_w], p_addr[16], p_data[27])
        p_fifo_w = 2 + self.tidx_w + 16 + 27
        prog_fifo = AsyncFIFO(
            width=p_fifo_w,
            depth=8,
            w_domain="host",
            r_domain="core",
        )
        m.submodules.prog_fifo = prog_fifo

        prog_wdata = Signal(p_fifo_w)
        prog_wen = Signal()

        with m.If(self.prog_route_en):
            m.d.comb += [
                prog_wdata.eq(Cat(
                    Const(0, 2),
                    self.prog_tile_idx,
                    self.prog_route_addr.as_unsigned()[:16],
                    self.prog_route_data.as_unsigned()[:27],
                )),
                prog_wen.eq(1),
            ]
        with m.Elif(self.prog_cb_en):
            m.d.comb += [
                prog_wdata.eq(Cat(
                    Const(1, 2),
                    self.prog_tile_idx,
                    self.prog_cb_addr.as_unsigned()[:16],
                    self.prog_cb_data.as_unsigned()[:27],
                )),
                prog_wen.eq(1),
            ]
        with m.Elif(self.prog_rail_en):
            m.d.comb += [
                prog_wdata.eq(Cat(
                    Const(2, 2),
                    self.prog_tile_idx,
                    self.prog_rail_addr.as_unsigned()[:16],
                    self.prog_rail_data.as_unsigned()[:27],
                )),
                prog_wen.eq(1),
            ]
        with m.Else():
            m.d.comb += prog_wen.eq(0)

        m.d.comb += [
            prog_fifo.w_data.eq(prog_wdata),
            prog_fifo.w_en.eq(prog_wen & prog_fifo.w_rdy),
        ]

        # In core domain: pop and apply write strobe atomically
        p_type = prog_fifo.r_data[0:2]
        p_tile = prog_fifo.r_data[2:2 + self.tidx_w]
        p_addr = prog_fifo.r_data[2 + self.tidx_w:18 + self.tidx_w]
        p_data = prog_fifo.r_data[18 + self.tidx_w:]

        m.d.comb += [
            prog_fifo.r_en.eq(prog_fifo.r_rdy),
            inner.prog_tile_idx.eq(p_tile),
            inner.prog_route_addr.eq(p_addr[:max(1, (self.route_depth - 1).bit_length())]),
            inner.prog_route_data.eq(p_data[:16]),
            inner.prog_cb_addr.eq(p_addr[:16]),
            inner.prog_cb_data.eq(p_data[:27]),
            inner.prog_rail_addr.eq(p_addr[:max(1, (self.rails - 1).bit_length())]),
            inner.prog_rail_data.eq(p_data[:8].as_signed()),
            inner.prog_route_en.eq(prog_fifo.r_rdy & (p_type == 0)),
            inner.prog_cb_en.eq(prog_fifo.r_rdy & (p_type == 1)),
            inner.prog_rail_en.eq(prog_fifo.r_rdy & (p_type == 2)),
        ]

        return m
