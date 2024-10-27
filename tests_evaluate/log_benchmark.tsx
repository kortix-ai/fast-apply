import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ArrowDown, ArrowUp, Minus } from 'lucide-react'

const metrics = [
    {
        name: 'Fiable Accuracy (evaluated by DeepSeek)',
        v16_1p5B: '98.75%',
        v16_7B: '99.95%',
        improved: true,
        highlight: true
    },
    {
        name: 'Original Accuracy (fully match Claude Output)',
        v16_1p5B: '66.00%',
        v16_7B: '76.00%',
        improved: true
    },
    {
        name: 'Sorted Accuracy',
        v16_1p5B: '79.00%',
        v16_7B: '90.00%',
        improved: true
    },
    {
        name: 'Avg Total Diff',
        v16_1p5B: '4.04',
        v16_7B: '9.31',
        improved: false
    },
    {
        name: 'Avg Total Diff (Sorted)',
        v16_1p5B: '1.67',
        v16_7B: '7.87',
        improved: false
    },
    {
        name: 'Avg Added Lines',
        v16_1p5B: '1.61',
        v16_7B: '7.29',
        improved: false
    },
    {
        name: 'Avg Added Lines (Sorted)',
        v16_1p5B: '0.52',
        v16_7B: '6.63',
        improved: false
    }
]

export default function Component() {
    return (
        <Card className="w-full max-w-4xl mx-auto">
            <CardHeader>
                <CardTitle className="text-2xl font-bold text-center">Model Performance Comparison</CardTitle>
            </CardHeader>
            <CardContent>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead className="w-[50%]">Metric</TableHead>
                            <TableHead className="text-center">1.5B Model</TableHead>
                            <TableHead className="text-center">7B Model</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {metrics.map((metric) => {
                            const v1 = parseFloat(metric.v16_1p5B)
                            const v2 = parseFloat(metric.v16_7B)
                            const betterValue = metric.improved ? (v2 > v1 ? 'v16_7B' : 'v16_1p5B') : (v2 < v1 ? 'v16_7B' : 'v16_1p5B')

                            return (
                                <TableRow key={metric.name}>
                                    <TableCell className="font-medium">{metric.name}</TableCell>
                                    <TableCell className="text-center">
                                        <span className={metric.highlight ? 'text-green-600 font-medium' : ''}>
                                            {metric.v16_1p5B}
                                        </span>
                                        {!metric.name.startsWith('Avg') && renderComparisonIcon(betterValue === 'v16_1p5B', metric.improved)}
                                    </TableCell>
                                    <TableCell className="text-center">
                                        <span className={metric.highlight ? 'text-green-600 font-medium' : ''}>
                                            {metric.v16_7B}
                                        </span>
                                        {!metric.name.startsWith('Avg') && renderComparisonIcon(betterValue === 'v16_7B', metric.improved)}
                                    </TableCell>
                                </TableRow>
                            )
                        })}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    )
}

function renderComparisonIcon(isBetter: boolean, improved: boolean) {
    if (!isBetter) return null
    return improved ? (
        <ArrowUp className="inline-block w-4 h-4 ml-1 text-green-600" />
    ) : (
        <ArrowDown className="inline-block w-4 h-4 ml-1 text-green-600" />
    )
}